from yacs.config import CfgNode as CN

from loguru import logger

import sys
import os

sys.path.append("")

### CONFIGS ###
hparams = CN()

# General settings
hparams.EXP_NAME = "test"
hparams.MODEL_DIR = f"experiments/{hparams.EXP_NAME}/model"
hparams.LOG_DIR = f"experiments/{hparams.EXP_NAME}/logs"
hparams.CONFIG_PATH = f"experiments/{hparams.EXP_NAME}"
hparams.MODEL_ARCH = "fcnet"
hparams.DATASET = "CIFAR10"
hparams.DS_DIRECTORY = "datasets"
hparams.SEED_VALUE = 42

# Hardware
hparams.HARDWARE = CN()
hparams.HARDWARE.NUM_WORKERS = 4

# Training process hparams
hparams.TRAINING = CN()
hparams.TRAINING.MAX_EPOCHS = 100
hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH = 1
hparams.TRAINING.DROPOUT_P = 0.1
hparams.TRAINING.VAL_SPLIT = 0.0
hparams.TRAINING.LR = 0.01
hparams.TRAINING.WD = 0.0001
hparams.TRAINING.MOMENTUM = 0.9
hparams.TRAINING.BATCH_SIZE = 64
hparams.TRAINING.AUGMENT = False
hparams.TRAINING.LR_DECAY = 0.1
hparams.TRAINING.DECAY_EPOCH = [60, 90]
hparams.TRAINING.NORMALIZE = False
hparams.TRAINING.WMLR = 0.01
hparams.TRAINING.WMWD = 0.0001
hparams.TRAINING.PRE_MIRROR = 2
hparams.TRAINING.MIRROR = 100
hparams.TRAINING.MODE = "modulated"

# PEPITA parameters
hparams.PEPITA = CN()
hparams.PEPITA.B_INIT = "normal"
hparams.PEPITA.B_MEAN_ZERO = True
hparams.PEPITA.BSTD = 0.05

# Model parameters
hparams.MODEL = CN()

# Parameters for FCNet
hparams.MODEL.FCNet = CN()
hparams.MODEL.FCNet.HIDDEN_LAYER_SIZES = [1024]
hparams.MODEL.FCNet.LAYER_INIT = "he_normal"
hparams.MODEL.FCNet.NORMALIZATION = False


def get_hparams_defaults():
    """Get a yacs hparamsNode object with default values for my_project."""
    return hparams.clone()


def update_paths(hparams):
    """Update hparams paths

    Args:
        hparams (CfgNode): params to update
    """
    hparams.MODEL_DIR = f"experiments/{hparams.EXP_NAME}/model"
    hparams.LOG_DIR = f"experiments/{hparams.EXP_NAME}/logs"
    hparams.CONFIG_PATH = f"experiments/{hparams.EXP_NAME}"
    hparams.CONFIG_PATH = f"experiments/{hparams.EXP_NAME}"


def update_hparams(hparams_file):
    """Return an updated yacs hparamsNode

    Args:
        hparams_file (string): path to the .yaml file
    """
    hparams = get_hparams_defaults()
    hparams.merge_from_file(hparams_file)
    update_paths(hparams)
    logger.info(f"Loaded current configuration from {hparams.CONFIG_PATH}/config.yaml")
    return hparams.clone()


def load_exp_hparams(exp_name):
    """Handles loading of hparams for given experiment

    Args:
        exp_name (string): name of the experiment
    """
    hparams.EXP_NAME = exp_name
    update_paths(hparams)
    hparams_file = os.path.join(hparams.CONFIG_PATH, "config.yaml")
    return update_hparams(hparams_file)


def update_hparams_from_cfg(cfg):
    """Return an updated yacs hparamsNode

    Args:
        cfg (cfg): cfg with the updated hparams
    """
    hparams = get_hparams_defaults()
    hparams.merge_from_other_cfg(cfg)
    update_paths(hparams)
    return hparams.clone()

def update_hparams_from_dict(cfg_dict):
    """Return an updated yacs hparamsNode

    Args:
        cfg_dict (dict): dict with the updated hparams
    """
    hparams = get_hparams_defaults()
    cfg = hparams.load_cfg(str(cfg_dict))
    hparams.merge_from_other_cfg(cfg)
    update_paths(hparams)
    return hparams.clone()


def save_config(hparams):
    """Saves the current configuration as .yaml

    Args:
        hparams (CfgNode): hparams to save
    """
    with open(os.path.join(hparams.CONFIG_PATH, "config.yaml"), "w") as f:
        f.write(hparams.dump())
        logger.info(f"Saved current configuration at {hparams.CONFIG_PATH}/config.yaml")
