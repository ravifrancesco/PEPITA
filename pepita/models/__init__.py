from loguru import logger

from .FCnet import FCNet
from .ResFCNet import ResFCNet

from ..dataset import get_data_info

def modelpool(MODELNAME, hparams):
    r"""Return the training and testing dataloaders of the given dataset TODO change

    Args:
        DATANAME (string): the name of the dataset
        batchsize (int): the batchsize to be used
        val_split (float, optional): validation split (defaul is 0.2)
        augment (bool, optional): if True, data augmentation is performed on the training dataloader (default is False)
        num_workers (int, optional): the number of workers that is passed to the DataLoader class (default is 8)

    Return:
        Dataloders (DataLoader, DataLoader, DataLoader): training, validation and testing dataloaders of the given dataset
    """

    img_w, n_chan, n_classes = get_data_info(hparams.DATASET)

    if MODELNAME.lower() == 'fcnet':
        input_size = img_w*img_w*n_chan
        hidden_layers = hparams.MODEL.FCNet.HIDDEN_LAYER_SIZES
        layers = [input_size] + hidden_layers + [n_classes]
        model = FCNet(
                layers,
                B_mean_zero=hparams.PEPITA.B_MEAN_ZERO, 
                Bstd=hparams.PEPITA.BSTD,
                p=hparams.TRAINING.DROPOUT_P
            )
        return model, input_size, n_classes, True
    if MODELNAME.lower() == 'resfcnet':
        input_size = img_w*img_w*n_chan
        model = ResFCNet(
            input_size,
            n_classes,
            hparams.MODEL.ResFCNet.BLOCK_SIZES,
            block_depth=hparams.MODEL.ResFCNet.BLOCK_DEPTH,
            res_connect=hparams.MODEL.ResFCNet.RES_CONNECT,
            B_mean_zero=hparams.PEPITA.B_MEAN_ZERO, 
            Bstd=hparams.PEPITA.BSTD,
            p=hparams.TRAINING.DROPOUT_P
        )
        return model, input_size, n_classes, True
    else:
        logger.error(f'Model \'{MODELNAME.lower()}\' is not implemented yet')
        exit() 