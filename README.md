# GANs Can Play Lottery Tickets Too

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code for this paper [GANs Can Play Lottery Tickets Too](https://openreview.net/forum?id=1AoMhc_9jER).
## Overview

For a range of GANs, we can find matching subnetworks at 67%-74% sparsity. We observe that with or without pruning discriminator has a minor effect on the existence and quality of matching subnetworks, while the initialization used in the discriminator plays a significant role.

## Experiment Results

Iterative pruning results on SNGAN

![](https://github.com/VITA-Group/GAN-LTH/blob/main/Figs/result.png)

## Requirements

`pytorch==1.4.0`
`tensorflow-gpu=1.15.0`
`imageio`
`scikit-image`
`tqdm`
`tensorboardx`

## Command


### SNGAN

#### Generate Initial Weights
```
mkdir initial_weights
python generate_initial_weights.py --model sngan_cifar10
```
#### Prepare FID statistics

Download FID statistics files from [here](https://www.dropbox.com/sh/lau1g8n0moj1lmi/AACCBGqAhSsjCcpA78VND18ta?dl=0) to `fid_stat`. 

#### Baseline
```
python train.py --model sngan_cifar10 --exp_name sngan_cifar10 --init-path initial_weights
```
#### Iterative Magnitude Pruning on Generator (IMPG)
```
python train_impg.py --model sngan_cifar10 --exp_name sngan_cifar10 --init-path initial_weights 
```
#### Iterative Magnitude Pruning on Generator (IMPGD)
```
python train_impgd.py --model sngan_cifar10 --exp_name sngan_cifar10 --init-path initial_weights 
```
### Iterative Magnitude Pruning on Generator (IMPGDKD)

```
python train_impgd.py --model sngan_cifar10 --exp_name sngan_cifar10 --init-path initial_weights --use-kd-d
```

### CycleGAN

#### Generate initial weights

```
mkdir initial_weights
python generate_initial_weights.py
```

#### Download Data

```
./download_dataset DATASET_NAME
```
#### Baseline

```
python train.py --dataset DATASET_NAME --rand initial_weights --gpu GPU 
```

#### IMPG

```
python train_impg.py --dataset DATASET_NAME --rand initial_weights --gpu GPU --pretrain PRETRAIN
```

#### IMPGD

```
python train_impg.py --dataset DATASET_NAME --rand initial_weights --gpu GPU --pretrain PRETRAIN
```
