## Spatial Self-modulation on BigGAN

## Requirements

This repository starts from <a href="https://github.com/uoguelph-mlrg/instance_selection_for_gans">Instance Selection for GANs Official PyTorch Code</a>, and requirements are exactly same.

## Overview

## Preparing ImageNet-64x64

To train a BigGAN on ImageNet you will first need to construct an HDF5 dataset file for ImageNet (optional), compute Inception moments for calculating FID, and construct the image manifold for calculating Precision, Recall, Density, and Coverage. All can by done by modifying and running 
```
bash scripts/utils/prepare_data_imagenet_[res].sh
```
where [res] is substituted with the desired resolution (options are 64, 128, or 256). These scripts will assume that ImageNet is in a folder called `data` in the instance_selection_for_gans directory. Replace this with the filepath to your copy of ImageNet. 

## Training on ImageNet-64x64

```.bash
bash scripts/launch_SAGAN_res64_ch32_bs128_dstep_1_rr100.sh
```

## Results on ImageNet-64x64

## Citation

```
@article{devries2020instance,
  title={Instance Selection for GANs},
  author={DeVries, Terrance and Drozdzal, Michal and Taylor, Graham W},
  journal={Advances in Neural Information Processing Systems},
  year={2020}
}
```
