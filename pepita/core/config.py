from yacs.config import CfgNode as CN

### CONSTANTS ###
DATASET_DIR = {
    'CIFAR10': 'datasets',
    'CIFAR100': 'datasets'
}

### CONFIGS ###
hparams = CN()

# General settings
hparams.EXP_NAME = 'cifar10_fcnet_nodropout'
hparams.MODEL_DIR = f'experiments/{hparams.EXP_NAME}/model'
hparams.LOG_DIR = f'experiments/{hparams.EXP_NAME}/logs'
hparams.MODEL_ARCH = 'fcnet'
hparams.DATASET = 'CIFAR10'
hparams.SEED_VALUE = 42

# Hardware
hparams.HARDWARE = CN()
hparams.HARDWARE.NUM_WORKERS = 4

# Training process hparams
hparams.TRAINING = CN()
hparams.TRAINING.MAX_EPOCHS = 100
hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH = 1
hparams.TRAINING.DROPOUT_P = 0
hparams.TRAINING.VAL_SPLIT = 0.0
hparams.TRAINING.LR = 0.01
hparams.TRAINING.BATCH_SIZE = 64
hparams.TRAINING.AUGMENT = False
hparams.TRAINING.LR_DECAY = 0.1
hparams.TRAINING.DECAY_EPOCH = [60, 90]

# PEPITA parameters
hparams.PEPITA = CN()
hparams.PEPITA.B_MEAN_ZERO = True
hparams.PEPITA.BSTD = 0.05

# Model parameters
hparams.MODEL = CN()

# Parameters for FCNet
hparams.MODEL.FCNet = CN()
hparams.MODEL.FCNet.HIDDEN_LAYER_SIZES = [1024] 

def get_hparams_defaults():
    """Get a yacs hparamsNode object with default values for my_project."""
    return hparams.clone()

def update_hparams(hparams_file):
    """Return an updated yacs hparamsNode
    
    Args:
        hparams_file (string): path to the .yaml file
    """
    hparams = get_hparams_defaults()
    hparams.merge_from_file(hparams_file)
    return hparams.clone()

def update_hparams_from_dict(cfg_dict):
    """Return an updated yacs hparamsNode
    
    Args:
        cfg_dict (dict): dict with the updated hparams
    """
    hparams = get_hparams_defaults()
    cfg = hparams.load_cfg(str(cfg_dict))
    hparams.merge_from_other_cfg(cfg)
    return hparams.clone()
