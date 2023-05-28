# SLViT
Code for IJCAI 2023 paper 'SLViT: Scale-Wise Language-Guided Vision Transformer for Referring Image Segmentation'.

## Dataset
Follow instructions in the ./refer directory to set up subdirectories and download annotations. This directory is a git clone (minus two data files that we do not need) from the [refer](https://github.com/lichengunc/refer) public API.

Download images from [COCO](https://cocodataset.org/#download). Please use the first downloading link 2014 Train images, and extract the downloaded `train_2014.zip` file to `./refer/data/images/mscoco/images`.

## Enviorments
- python 3.7.0
- pytorch 1.7.1
- torchvision 0.8.2

## Reference
- [LAVT](https://github.com/yz93/LAVT-RIS)
- [SegNeXt](https://github.com/Visual-Attention-Network/SegNeXt)