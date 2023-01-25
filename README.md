# PEPITA

Example of usage:

* 1 Hidden layer: `python scripts/train.py -en test_1 -d cifar100 -ls 1024 -wd 0.0 -bi uniform -li he_uniform`
* 2 Hidden layers: `python scripts/train.py -en test_2 -d cifar100 -ls 1024 1024 -wd 0.0 -bi uniform -li he_uniform`
* 3 Hidden layers: `python scripts/train.py -en test_2 -d cifar100 -ls 1024 1024 1024 -wd 0.0 -bi uniform -li he_uniform`

Run `python scripts/train.py -h` to have an overview of the flags and their usage.
