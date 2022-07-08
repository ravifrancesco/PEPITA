from .getdataloader import *

def datapool(DATANAME, batchsize, augment=False, num_workers=8):
    r"""Return the training and testing dataloaders of the given dataset

    Args:
        DATANAME (string): the name of the dataset
        batchsize (int): the batchsize to be used
        augment (bool, optional): if True, data augmentation is performed on the training dataloader (default is False)
        num_workers (int, optional): the number of workers that is passed to the DataLoader class (default is 8)

    Raises:
        NotImplementedError: If the dataset name is not valid

    Returns:
        Dataloders (DataLoader, DataLoader): training and testing dataloaders of the given dataset
    """

    if DATANAME.lower() == 'cifar10':
        return get_cifar10(batchsize, augment=augment, num_workers=num_workers)
    elif DATANAME.lower() == 'cifar100':
        return get_cifar100(batchsize, augment=augment, num_workers=num_workers)
    else:
        raise NotImplementedError(f'Dataset \'{DATANAME.lower()}\' is not supported yet')