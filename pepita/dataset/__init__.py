from loguru import logger

from .getdataloader import *

def datapool(DATANAME, batchsize, val_split=0.2, augment=False, num_workers=8, normalize=False):
    r"""Return the training and testing dataloaders of the given dataset

    Args:
        DATANAME (string): the name of the dataset
        batchsize (int): the batchsize to be used
        val_split (float, optional): validation split (defaul is 0.2)
        augment (bool, optional): if True, data augmentation is performed on the training dataloader (default is False)
        num_workers (int, optional): the number of workers that is passed to the DataLoader class (default is 8)

    Return:
        Dataloders (DataLoader, DataLoader, DataLoader): training, validation and testing dataloaders of the given dataset
    """

    if DATANAME.lower() == 'mnist':
        return get_mnist(batchsize, val_split=val_split, augment=augment, num_workers=num_workers, normalize=normalize)
    elif DATANAME.lower() == 'cifar10':
        return get_cifar10(batchsize, val_split=val_split, augment=augment, num_workers=num_workers, normalize=normalize)
    elif DATANAME.lower() == 'cifar100':
        return get_cifar100(batchsize, val_split=val_split, augment=augment, num_workers=num_workers, normalize=normalize)
    else:
        logger.error(f'Dataset \'{DATANAME.lower()}\' is not supported yet')
        exit()

def get_data_info(DATANAME):
    r"""Return the image size, number of channels and number of classes

    Args:
        DATANAME (string): the name of the dataset

    Return:
        data info (int, int, int): image size (width in pixels) number of channels, number of classes
    """
    
    if DATANAME.lower() == 'mnist':
        return 28, 1, 10
    elif DATANAME.lower() == 'cifar10':
        return 32, 3, 10
    elif DATANAME.lower() == 'cifar100':
        return 32, 3, 100
    else:
        logger.error(f'Dataset \'{DATANAME.lower()}\' is not supported yet')
        exit()