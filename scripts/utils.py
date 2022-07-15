from yacs.config import CfgNode as CN

def create_arg_cfg(args):
    r"""Returns cfg given the arguments

    Args:
        cfg (CfgNode): arguments for creating the cfg
    """
    cfg = CN()
    cfg.EXP_NAME = args.exp_name
    cfg.MODEL_ARCH = args.arch
    cfg.DATASET = args.dataset
    cfg.SEED_VALUE = args.seed

    cfg.HARDWARE = CN()
    cfg.HARDWARE.NUM_WORKERS = args.workers

    cfg.TRAINING = CN()
    cfg.TRAINING.MAX_EPOCHS = args.epochs
    cfg.TRAINING.DROPOUT_P = args.dropout
    cfg.TRAINING.VAL_SPLIT = args.val_split
    cfg.TRAINING.LR = args.learning_rate
    cfg.TRAINING.BATCH_SIZE = args.batch_size
    cfg.TRAINING.AUGMENT = args.augment
    cfg.TRAINING.LR_DECAY = args.decay
    cfg.TRAINING.DECAY_EPOCH = args.decay_epoch

    cfg.PEPITA = CN()
    cfg.PEPITA.B_MEAN_ZERO = args.b_mean_zero
    cfg.PEPITA.BSTD = args.bstd

    return cfg