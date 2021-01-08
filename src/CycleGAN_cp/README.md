# GAN_Slimming_CycleGAN

## GAN slimming 3-in-1 framework:
Compress (GS-32):
```
python gs.py --rho 0.01
```

Compress (GS-8):
```
python gs.py --rho 0.01 --quant
```

Extract subnetwork obtained by channel pruning (CP):
```
python extract_subnet.py --model_str <model_str>
```

Finetune subnetwork:
```
python finetune.py --base_model_str <base_model_str>
```

## GS-32 + quantization aware finetune
Replace the finetune step in GS by the following (note that base model must be 32 bit model):
```
python finetune.py --base_model_str <base_model_str>
```

## Distillation alone (Huawei AAAI'20)
Distill (1/2 channel number student network):
```
python distill.py --alpha 0.5
```

## Channel pruning + distillation as finetune (Graph Adaptive Pruning. arXiv 2018)
CP:
```
python cp.py --rho 0.001
```

Extract subnetwork obtained by channel pruning (CP):
```
python extract_subnet.py --model_str <model_str>
```

Finetune subnetwork:
```
python finetune.py --base_model_str <base_model_str>
```

