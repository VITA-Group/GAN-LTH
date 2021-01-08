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
### Baseline

python main
## Iterative Magnitude Pruning on Generator (IMPG)

## Iterative Magnitude Pruning on Generator and Discriminator (IMPGD)
