# SLViT
Code for IJCAI 2023 paper 'SLViT: Scale-Wise Language-Guided Vision Transformer for Referring Image Segmentation'.

## Datasets
Refer to the instructions provided in the `./refer` directory to establish subdirectories and retrieve annotations. This directory contains a clone of the [refer](https://github.com/lichengunc/refer) public API, excluding two unnecessary data files.

Download images from [COCO](https://cocodataset.org/#download). Please use the first downloading link 2014 Train images, and extract the downloaded `train_2014.zip` file to `./refer/data/images/mscoco/images`.

## Enviorments
- python 3.7.0
- pytorch 1.7.1
- torchvision 0.8.2
- torchaudio 0.7.2
- mmcv-full 1.3.12
- mmsegmentation 0.17.0

## Reference
- [LAVT](https://github.com/yz93/LAVT-RIS)
- [SegNeXt](https://github.com/Visual-Attention-Network/SegNeXt)