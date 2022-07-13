import os
import sys
import torch
import argparse

from loguru import logger

import pytorch_lightning as pl

sys.path.append('')

from pepita.core.trainer import PEPITATrainer
from pepita.core.config import get_hparams_defaults

def main(hparams):

    log_dir = hparams.LOG_DIR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #TODO set_seed(hparams.SEED_VALUE)

    logger.add(
        os.path.join(log_dir, 'test.log'),
        level='INFO',
        colorize=False,
    )

    logger.info(f'Using device: {device}')
    logger.info(f'Hyperparameters: \n {hparams}')

    model = PEPITATrainer(hparams=hparams).to(device)

    
    logger.warning(f'Loading pretrained model from {hparams.MODEL_DIR}/model.ckpt')
    model.load_from_checkpoint('{hparams.MODEL_DIR}/model.ckpt')

    # most basic trainer, uses good defaults (1 gpu)
    trainer = pl.Trainer(
        #gpus=1,
        resume_from_checkpoint=hparams.TRAINING.RESUME,
        logger=None,
    )

    logger.info('*** Started testing ***')
    trainer.test(model=model)
    logger.info('*** Test Ended ***')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()

    logger.info(f'Input arguments: \n {args}')

    hparams = get_hparams_defaults()

    main(hparams)