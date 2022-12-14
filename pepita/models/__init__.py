from loguru import logger

from .FCnet import FCNet
from .ConvNet import ConvNet

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
        return model, (input_size, ), n_classes, True

    elif MODELNAME.lower() == "convnet":
        hidden_layers = hparams.MODEL.FCNet.HIDDEN_LAYER_SIZES  # FIXME same for conv
        fc_layer_size = 4096  # FIXME this should be an extra parameter, or could be inferred from img size
        conv_layers_channels = [n_chan] + hidden_layers
        model = ConvNet(
            conv_channels=conv_layers_channels,
            fc_layer_size=fc_layer_size,
            n_classes=n_classes,
            img_shape=(img_w, img_w),
            fc_dropout_p=hparams.TRAINING.DROPOUT_P,
            init=hparams.MODEL.FCNet.LAYER_INIT,  # FIXME same for conv
            B_init=hparams.PEPITA.B_INIT,
            B_mean_zero=hparams.PEPITA.B_MEAN_ZERO, 
            Bstd=hparams.PEPITA.BSTD,
        )
        return model, (n_chan, img_w, img_w), n_classes, False

    else:
        logger.error(f'Model \'{MODELNAME.lower()}\' is not implemented yet')
        exit() 
