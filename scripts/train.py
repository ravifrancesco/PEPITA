from distutils.log import info
import os
import sys
import torch
import argparse

from loguru import logger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append('')

from pepita.core.trainer import PEPITATrainer
from pepita.core.config import get_hparams_defaults

def main(hparams, fast_dev_run=False):
    log_dir = hparams.LOG_DIR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #set_seed(hparams.SEED_VALUE)

    logger.add(
        os.path.join(log_dir, 'train.log'),
        level='INFO',
        colorize=False,
    )

    logger.info(f'Using device: {device}')
    logger.info(f'Hyperparameters: \n {hparams}')

    experiment_loggers = []

    # initialize tensorboard logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name='tb_logs',
        log_graph=True,
    )

    experiment_loggers.append(tb_logger)

    model = PEPITATrainer(hparams=hparams).to(device)

    ckpt_callback = ModelCheckpoint(
        monitor='validation_loss',
        save_last=True,
        verbose=True,
        save_top_k=1,
        mode='min',
        dirpath=hparams.MODEL_DIR,
        filename="model-{epoch:02d}-{val_loss:.2f}"
    )

    trainer = pl.Trainer(
        #cgpus=1,
        logger=experiment_loggers,
        max_epochs=hparams.TRAINING.MAX_EPOCHS,
        log_every_n_steps=50,
        #enable_checkpointing=ckpt_callback,
        #checkpoint_callback=ckpt_callback,
        #callbacks=[ckpt_callback],
        #terminate_on_nan=True,
        default_root_dir=log_dir,
        #progress_bar_refresh_rate=50,
        check_val_every_n_epoch=hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH,
        #resume_from_checkpoint=hparams.TRAINING.RESUME,
        num_sanity_val_steps=0,
        fast_dev_run=fast_dev_run,
        limit_val_batches=0 if not hparams.TRAINING.VAL_SPLIT else 1
    )

    logger.info('*** Started training ***')
    trainer.fit(model)
    logger.info('*** Training Ended ***')
    trainer.save_checkpoint(f'{hparams.MODEL_DIR}/model.ckpt')
    logger.info(f'Model saved at {hparams.MODEL_DIR}/model.ckpt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--fdr', action='store_true', help='fast dev run')

    args = parser.parse_args()

    logger.info(f'Input arguments: \n {args}')

    hparams = get_hparams_defaults()

    main(hparams, fast_dev_run=args.fdr)