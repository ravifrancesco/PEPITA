import os
import sys
import torch
import argparse

from loguru import logger

import pytorch_lightning as pl

sys.path.append('')

from pepita.core.trainer import PEPITATrainer
from pepita.core.config import load_exp_hparams

def main(hparams):

    log_dir = hparams.LOG_DIR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.add(
        os.path.join(log_dir, 'test.log'),
        level='INFO',
        colorize=False,
    )

    logger.info(f'Using device: {device}')
    logger.info(f'Hyperparameters: \n {hparams}')

    logger.warning(f'Loading pretrained model from {hparams.MODEL_DIR}/model.ckpt')
    model = PEPITATrainer.load_from_checkpoint(f'{hparams.MODEL_DIR}/model.ckpt', hparams=hparams).to(device)

    # most basic trainer, uses good defaults (1 gpu)
    trainer = pl.Trainer(
        #gpus=1,
        #resume_from_checkpoint=hparams.TRAINING.RESUME,
        logger=None,
    )

    logger.info('*** Started testing ***')
    trainer.test(model=model)
    logger.info('*** Test Ended ***')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-en', '--exp_name', type=str, help='experiment name', required=True)

    args = parser.parse_args()

    hparams = load_exp_hparams(args.exp_name)

    main(hparams)
