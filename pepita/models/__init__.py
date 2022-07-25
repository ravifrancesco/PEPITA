from loguru import logger

from pepita.models.SkipFCNet import SkipFCNet

from .FCnet import FCNet
from .SkipFCNet import SkipFCNet
from .TestNet import TestNet
from .TestNet2 import TestNet2
from .FCNetMirror import FCNetMirror


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
                init=hparams.MODEL.FCNet.LAYER_INIT,
                B_init=hparams.PEPITA.B_INIT,
                B_mean_zero=hparams.PEPITA.B_MEAN_ZERO, 
                Bstd=hparams.PEPITA.BSTD,
                p=hparams.TRAINING.DROPOUT_P
            )
        return model, input_size, n_classes, True
    if MODELNAME.lower() == 'skipfcnet':
        input_size = img_w*img_w*n_chan
        model = SkipFCNet(
            input_size,
            n_classes,
            hparams.MODEL.SkipFCNet.BLOCK_SIZES,
            B_mean_zero=hparams.PEPITA.B_MEAN_ZERO, 
            Bstd=hparams.PEPITA.BSTD,
            p=hparams.TRAINING.DROPOUT_P
        )
        return model, input_size, n_classes, True
    if MODELNAME.lower() == 'testnet':
        input_size = img_w*img_w*n_chan
        hidden_layers = hparams.MODEL.FCNet.HIDDEN_LAYER_SIZES
        layers = [input_size] + hidden_layers + [n_classes]
        model = TestNet(
                layers,
                init=hparams.MODEL.FCNet.LAYER_INIT,
                B_init=hparams.PEPITA.B_INIT,
                B_mean_zero=hparams.PEPITA.B_MEAN_ZERO, 
                Bstd=hparams.PEPITA.BSTD,
                p=hparams.TRAINING.DROPOUT_P,
                b_decay=hparams.MODEL.TestNet.B_DECAY
            )
        return model, input_size, n_classes, True
    if MODELNAME.lower() == 'testnet2':
        input_size = img_w*img_w*n_chan
        hidden_layers = hparams.MODEL.FCNet.HIDDEN_LAYER_SIZES
        layers = [input_size] + hidden_layers + [n_classes]
        model = TestNet2(
                layers,
                init=hparams.MODEL.FCNet.LAYER_INIT,
                B_init=hparams.PEPITA.B_INIT,
                B_mean_zero=hparams.PEPITA.B_MEAN_ZERO, 
                Bstd=hparams.PEPITA.BSTD,
                p=hparams.TRAINING.DROPOUT_P,
                b_decay=hparams.MODEL.TestNet.B_DECAY
            )
        return model, input_size, n_classes, True
    if MODELNAME.lower() == 'fcnetmirror':
        input_size = img_w*img_w*n_chan
        hidden_layers = hparams.MODEL.FCNet.HIDDEN_LAYER_SIZES
        layers = [input_size] + hidden_layers + [n_classes]
        model = FCNetMirror(
                layers,
                init=hparams.MODEL.FCNet.LAYER_INIT,
                B_init=hparams.PEPITA.B_INIT,
                B_mean_zero=hparams.PEPITA.B_MEAN_ZERO, 
                Bstd=hparams.PEPITA.BSTD,
                p=hparams.TRAINING.DROPOUT_P
            )
        return model, input_size, n_classes, True
    else:
        logger.error(f'Model \'{MODELNAME.lower()}\' is not implemented yet')
        exit() 