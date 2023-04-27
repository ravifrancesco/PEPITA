# PEPITA

To download the repository locally, please download the file `PEPITA.zip`

## Example of usage

* 1 Hidden layer: `python scripts/train.py -en test_1 -d cifar100 -ls 1024 -wd 0.0 -bi uniform -li he_uniform`
* 2 Hidden layers: `python scripts/train.py -en test_2 -d cifar100 -ls 1024 1024 -wd 0.0 -bi uniform -li he_uniform`
* 3 Hidden layers: `python scripts/train.py -en test_2 -d cifar100 -ls 1024 1024 1024 -wd 0.0 -bi uniform -li he_uniform`

Run `python scripts/train.py -h` to have an overview of the flags and their usage.

All experiments are logged by `tensorboard` in the `experiments` folder.

## Requirements

The following libraries are required for running the code. The indicated versions are the one we used, but other versions may work.
```
torch==1.13.1
torchvision==0.14.1
pytorch_lightning==1.6.5
torchmetrics==0.9.2

loguru==0.6.0
numpy==1.21.2
pillow==8.4.0
scikit-learn==1.2.1
```

Additionally, `ray[tune]` is needed for grid searches.
