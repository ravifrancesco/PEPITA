from yacs.config import CfgNode as CN

### CONSTANTS ###
DATASET_DIR = {
    'CIFAR10': 'E:\datasets',
    'CIFAR100': 'E:\datasets'
}

### CONFIGS ###
hparams = CN()

# General settings
hparams.MODEL_DIR = 'models/experiments'
hparams.LOG_DIR = 'logs/experiments'
hparams.MODEL = 'fcnet'
hparams.EXP_NAME = 'default'
hparams.PROJECT_NAME = 'fcnet'
hparams.SEED_VALUE = 42

# Hardware
hparams.HARDWARE = CN()
hparams.HARDWARE.NUM_WORKERS = 8

# Training process hparams
hparams.TRAINING = CN()
hparams.TRAINING.RESUME = None
hparams.TRAINING.PRETRAINED = None
hparams.TRAINING.PRETRAINED_LIT = None
hparams.TRAINING.MAX_EPOCHS = 100
hparams.TRAINING.LOG_SAVE_INTERVAL = 50
hparams.TRAINING.LOG_FREQ_TB_IMAGES = 500
hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH = 1
hparams.TRAINING.DROPOUT_P = 0.2
hparams.TRAINING.VAL_SPLIT = 0.2
hparams.TRAINING.BATCH_SIZE = 32
hparams.TRAINING.AUGMENT = False

# PEPITA parameters
hparams.PEPITA = CN()
hparams.PEPITA.B_MEAN_ZERO = True
hparams.PEPITA.BSTD = 0.05

# Model parameters
hparams.MODEL = CN()

# Parameters for FCNet
hparams.MODEL.FCNet = CN()
hparams.MODEL.FCNet.HIDDEN_LAYER_SIZES = [1024, 256] 

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
