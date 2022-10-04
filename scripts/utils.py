from yacs.config import CfgNode as CN

from ray import tune

import random
import sys

def create_arg_cfg(args):
    r"""Returns cfg given the arguments

    Args:
        args (Namespace): arguments for creating the cfg
    """
    cfg = CN()
    cfg.EXP_NAME = args.exp_name
    cfg.MODEL_ARCH = args.arch
    cfg.DATASET = args.dataset
    cfg.SEED_VALUE = args.seed if args.seed else random.randint(0, 4294967295)

    cfg.HARDWARE = CN()
    cfg.HARDWARE.NUM_WORKERS = args.workers

    cfg.TRAINING = CN()
    cfg.TRAINING.MAX_EPOCHS = args.epochs
    cfg.TRAINING.DROPOUT_P = args.dropout
    cfg.TRAINING.VAL_SPLIT = args.val_split
    cfg.TRAINING.LR = args.learning_rate
    cfg.TRAINING.WD = args.weight_decay
    cfg.TRAINING.MOMENTUM = args.momentum
    cfg.TRAINING.BATCH_SIZE = args.batch_size
    cfg.TRAINING.AUGMENT = args.augment
    cfg.TRAINING.LR_DECAY = args.decay
    cfg.TRAINING.DECAY_EPOCH = args.decay_epoch
    cfg.TRAINING.WMLR = args.wm_learning_rate
    cfg.TRAINING.WMWD = args.wm_weight_decay
    cfg.TRAINING.PRE_MIRROR = args.pre_mirror
    cfg.TRAINING.MIRROR = args.mirror

    cfg.PEPITA = CN()
    cfg.PEPITA.B_INIT = args.b_init
    cfg.PEPITA.B_MEAN_ZERO = args.b_mean_zero
    cfg.PEPITA.BSTD = args.bstd

    return cfg

def create_grid_search_dict(args):
    r"""Returns dict given the arguments

    Args:
        args (Namespace): arguments for creating the dict
    """
    return {
        "EXP_NAME" : args.exp_name,
        "MODEL_ARCH" : args.arch,
        "DATASET" : args.dataset,
        "SEED_VALUE" : args.seed if args.seed else random.randint(0, 4294967295),

        "HARDWARE" : {
            "NUM_WORKERS" : args.workers
        },

        "TRAINING" : {
            "MAX_EPOCHS" : args.epochs,
            "DROPOUT_P" : tune.grid_search(args.dropout),
            "VAL_SPLIT" : args.val_split,
            "LR" : tune.grid_search(args.learning_rate),
            "WD" : tune.grid_search(args.weight_decay),
            "MOMENTUM" : tune.grid_search(args.momentum),
            "BATCH_SIZE" : args.batch_size,
            "AUGMENT" : args.augment,
            "LR_DECAY" : tune.grid_search(args.decay),
            "DECAY_EPOCH" : tune.grid_search(args.decay_epoch),
            "WMLR" : tune.grid_search(args.wm_learning_rate),
            "WMWD" : tune.grid_search(args.wm_weight_decay),
            "PRE_MIRROR" : tune.grid_search(args.pre_mirror),
            "MIRROR" : tune.grid_search(args.mirror),
        },

        "PEPITA" : {
            "B_INIT" : args.b_init,
            "B_MEAN_ZERO" : args.b_mean_zero,
            "BSTD" : tune.grid_search(args.bstd)
        }
    }