from .getdataloader import *

def datapool(DATANAME, batchsize, augment=False, num_workers=8):
    if DATANAME.lower() == 'cifar10':
        return get_cifar10(batchsize, augment=augment, num_workers=num_workers)
    elif DATANAME.lower() == 'cifar100':
        return GetCifar100(batchsize, augment=augment, num_workers=num_workers)
    else:
        print(f'{DATANAME.lower()} is not supported yet')
        exit(-1)