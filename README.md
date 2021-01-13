# GanTicketsICLR

## Requirements

`pytorch==1.4.0`
`tensorflow-gpu=1.15.0`
`imageio`
`scikit-image`
`tqdm`
`tensorboardx`

## Command

## Generate Initial Weights
mkdir initial_weights
python generate_initial_weights.py --model sngan_cifar10

## Prepare FID statistics

Download fid_stat:
### Baseline

python train.py --model sngan_cifar10 --exp_name sngan_cifar10 --init-path initial_weights
## Iterative Magnitude Pruning on Generator (IMPG)

python train_impg.py --model sngan_cifar10 --exp_name sngan_cifar10 --init-path initial_weights --load-path `path_to_checkpoint` 


## Iterative Magnitude Pruning on Generator and Discriminator (IMPGD)
