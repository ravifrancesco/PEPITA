from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
from Preprocess.augment import Cutout, CIFAR10Policy

# code adapted from https://github.com/putshua/SNN_conversion_QCFS/blob/master/Preprocess/getdataloader.py

DIR = {'CIFAR10': 'E:\datasets', 'CIFAR100': 'E:\datasets'}

def get_cifar10(batch_size, augment=False, num_workers=8):
    
    if augment:
        trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      CIFAR10Policy(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                      Cutout(n_holes=1, length=16)
                                    ])
    else:
        trans_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=num_workers)

    return train_dataloader, test_dataloader

def get_cifat100(batch_size, augment=False, num_workers=8):

    if augment:
        trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
                                    Cutout(n_holes=1, length=16)
                                    ])
    else:
        trans_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])])
    
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])])

    train_data = datasets.CIFAR100(DIR['CIFAR100'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR100(DIR['CIFAR100'], train=False, transform=trans, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, test_dataloader