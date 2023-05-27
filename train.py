import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from functools import reduce
import operator
from bert.modeling_bert import BertModel

import torchvision
from lib import segmentation

import transforms as T
import utils
import numpy as np

import torch.nn.functional as F
from torch.autograd import Variable

import gc
from collections import OrderedDict
from thop import profile


def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None
                      )
    num_classes = 2

    return ds, num_classes


# IoU calculation for validation
def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            # print(logpt, at, logpt.shape, at.shape)
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def criterion(input, target):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    # FL = FocalLoss(gamma=2, alpha=0.55)
    # l1 = FL(input, target)
    # l2 = nn.functional.cross_entropy(input, target, weight=weight)
    # print(l1, l2)
    # return l1*10
    # return (l1*10 + l2)/2
    return nn.functional.cross_entropy(input, target, weight=weight)


def evaluate(model, data_loader, bert_model):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                                   target.cuda(non_blocking=True),\
                                                   sentences.cuda(non_blocking=True),\
                                                   attentions.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
                output = model(image, embedding, l_mask=attentions)
            else:
                output = model(image, sentences, l_mask=attentions)

            iou, I, U = IoU(output, target)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return 100 * iou, 100 * cum_I / cum_U


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0
    # print number of parameter
    totalP = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (totalP/1e6))
    # print(model)

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, attentions = data
        image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                               target.cuda(non_blocking=True),\
                                               sentences.cuda(non_blocking=True),\
                                               attentions.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        if bert_model is not None:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
            # print size of model
            # flops, params = profile(model, (image, embedding,))
            # print('flops: ', flops, 'params: ', params)
            output = model(image, embedding, l_mask=attentions)
        else:
            # print size of model
            # flops, params = profile(model, (image, sentences, attentions,))
            # print("flops: %.2fG" % (flops/1e9))
            # print("params: %.2fM" % (params/1e6))
            output = model(image, sentences, l_mask=attentions)

        loss = criterion(output, target)
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main(args):
    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)
    dataset_test, _ = get_dataset("val",
                                  get_transform(args=args),
                                  args=args)

    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    # model initialization
    print(args.model)
    # model = torch.load('./checkpoints/model_best_qfmid1_refcoco.pth')

    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                              args=args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = model.module

    if args.model != 'slvit':
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
        bert_model.cuda()
        bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
        single_bert_model = bert_model.module
    else:
        bert_model = None
        single_bert_model = None
    

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        if args.model != 'slvit':
            single_bert_model.load_state_dict(checkpoint['bert_model'])
    # checkpoint = torch.load('./checkpoints/model_best_mscan_mid_ham20_refcocog.pth', map_location='cpu')
    # single_model.load_state_dict(checkpoint['model'])
    
    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    if args.model != 'slvit':
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_bert_model.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]

    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    # training loops
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations, bert_model)
        iou, overallIoU = evaluate(model, data_loader_test, bert_model)

        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        save_checkpoint = (best_oIoU < overallIoU)
        if save_checkpoint:
            print('Better epoch: {}\n'.format(epoch))
            if single_bert_model is not None:
                dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}
            else:
                dict_to_save = {'model': single_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}

            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_mscan_mid_ham21_{}.pth'.format(args.model_id)))
            best_oIoU = overallIoU

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
