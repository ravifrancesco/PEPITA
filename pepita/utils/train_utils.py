import os

import pytorch_lightning as pl
import torch
import random

from loguru import logger

def seed_everything(seed_value):
    r"""Set experiment seed

    Args:
        seed_value (int): seed value
    """
    if seed_value >= 0:
        logger.warning(f'Seed value for the experiment {seed_value}')
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        pl.trainer.seed_everything(seed_value)
    else:
        logger.error(f'Value {seed_value} is not valid: seed value mustbe positive')
        exit()