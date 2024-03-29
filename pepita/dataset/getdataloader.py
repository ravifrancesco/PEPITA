import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split

from loguru import logger

from .augment import Cutout, CIFAR10Policy

def train_val_dataset(dataset, val_split=0.2):
    r"""Splits the input dataset into train and validation datasets

    Args:
        dataset (torch.utils.data.Dataset): dataset to split
        val_split (float, optional): validation split (defaul is 0.2)
    
    Returns:
        (Dataset, Dataset): training and validation datasets
    """
    if val_split == 0:
        return dataset, None
    
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    return train_dataset, val_dataset

# code adapted from https://github.com/putshua/SNN_conversion_QCFS/blob/master/Preprocess/getdataloader.py

def get_mnist(batchsize, val_split=0.2, augment=False, num_workers=8, normalize=False, ds_directory='datasets'): # TODO DOC and change
    
    normalizer = (transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),) if normalize else ()

    if augment:
        trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      CIFAR10Policy(),
                                      transforms.ToTensor(),
                                      *normalizer,
                                      Cutout(n_holes=1, length=16)
                                    ])
    else:
        trans_t = transforms.Compose([transforms.ToTensor(), *normalizer])
    
    trans = transforms.Compose([transforms.ToTensor(), *normalizer])

    train_data = datasets.MNIST(ds_directory, train=True, transform=trans_t, download=True)
    test_data = datasets.MNIST(ds_directory, train=False, transform=trans, download=True)
    train_data, val_data = train_val_dataset(train_data, val_split=val_split)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True)

    logger.info('Loaded MNIST dataset')

    return train_dataloader, val_dataloader, test_dataloader

def get_cifar10(batchsize, val_split=0.2, augment=False, num_workers=8, normalize=False, ds_directory='datasets'):
    r"""Return the training and testing dataloaders for CIFAR10

    Args:
        batchsize (int): the batchsize to be used
        val_split (float, optional): validation split (defaul is 0.2)
        augment (bool, optional): if true, data augmentation is performed on the training dataloader (default is False)
        num_workers (int, optional): the number of workers that is passed to the DataLoader class (default is 8)
        normalize (bool, optional): if True, normalization is applied (defaul is False)

    Returns:
        (DataLoader, DataLoader, DataLoader): training, validation and testing dataloaders for CIFAR10
    """

    normalizer = (transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),) if normalize else ()

    if augment:
        trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      CIFAR10Policy(),
                                      transforms.ToTensor(),
                                      *normalizer,
                                      Cutout(n_holes=1, length=16)
                                    ])
    else:
        trans_t = transforms.Compose([transforms.ToTensor(), *normalizer])
    
    trans = transforms.Compose([transforms.ToTensor(), *normalizer])

    train_data = datasets.CIFAR10(ds_directory, train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10(ds_directory, train=False, transform=trans, download=True)
    train_data, val_data = train_val_dataset(train_data, val_split=val_split)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True)

    logger.info('Loaded CIFAR10 dataset')

    return train_dataloader, val_dataloader, test_dataloader

def get_cifar100(batchsize, val_split=0.2, augment=False, num_workers=8, normalize=False, ds_directory='datasets'):
    r"""Return the training and testing dataloaders for CIFAR100

    Args:
        batchsize (int): the batchsize to be used
        val_split (float, optional): validation split (defaul is 0.2)
        augment (bool, optional): if true, data augmentation is performed on the training dataloader (default is False)
        num_workers (int, optional): the number of workers that is passed to the DataLoader class (default is 8)
        normalize (bool, optional): if True, normalization is applied (defaul is False)

    Returns:
        (DataLoader, DataLoader, DataLoader): training, validation and testing dataloaders for CIFAR100
    """

    normalizer = (transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),) if normalize else ()

    if augment:
        trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    *normalizer,
                                    Cutout(n_holes=1, length=16)
                                    ])
    else:
        trans_t = transforms.Compose([transforms.ToTensor(), *normalizer])
    
    trans = transforms.Compose([transforms.ToTensor(), *normalizer])

    train_data = datasets.CIFAR100(ds_directory, train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR100(ds_directory, train=False, transform=trans, download=True)
    train_data, val_data = train_val_dataset(train_data, val_split=val_split)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True)

    logger.info('Loaded CIFAR100 dataset')

    return train_dataloader, val_dataloader, test_dataloader