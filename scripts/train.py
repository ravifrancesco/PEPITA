from distutils.command import check
import os
import sys
from turtle import update
import torch
import argparse

from loguru import logger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append('')

from pepita.core.trainer import PEPITATrainer
from pepita.core.config import get_hparams_defaults, update_hparams, update_hparams_from_cfg, save_config
from pepita.utils.train_utils import seed_everything
from utils import create_arg_cfg

def main(hparams, fast_dev_run=False, checkpoint=False):
    
    log_dir = hparams.LOG_DIR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    seed_everything(hparams.SEED_VALUE)

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

    model = PEPITATrainer(hparams=hparams)

    ckpt_callback = ModelCheckpoint(
        monitor='validation_loss',
        save_last=True,
        verbose=True,
        save_top_k=1,
        mode='min',
        dirpath=hparams.MODEL_DIR,
        filename="model-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = pl.Trainer(
        #cgpus=1,
        logger=experiment_loggers,
        max_epochs=hparams.TRAINING.MAX_EPOCHS,
        log_every_n_steps=50,
        checkpoint_callback=checkpoint,
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
        progress_bar_refresh_rate=0,
        #limit_val_batches=0 if not hparams.TRAINING.VAL_SPLIT else 1 FIXME change
        gpus=1 if device=='cuda' else 0,
    )

    save_config(hparams)

    logger.info('*** Started training ***')
    trainer.fit(model)
    logger.info('*** Training Ended ***')
    if (checkpoint):
        trainer.save_checkpoint(f'{hparams.MODEL_DIR}/model.ckpt')
        logger.info(f'Model saved at {hparams.MODEL_DIR}/model.ckpt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', '--config_file', type=str, help='cfg file path')
    parser.add_argument('-fdr', '--fast_dev_run', action='store_true', help='fast dev run')
    parser.add_argument('-en', '--exp_name', type=str, default='exp', help="Experiment name")
    parser.add_argument('-ckp', '--checkpoint', action='store_true', help="Save checkpoint at the end of training")
    parser.add_argument('-a', '--arch', type=str, default='fcnet', help="Model architecture")
    parser.add_argument('-ls', '--layer_sizes', nargs='*', type=int, default=[1024], help="Sizes of the hidden layers")
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', help="Dataset")
    parser.add_argument('-ddir', '--ds_directory', type=str, help="Dataset directory", default='datasets')
    parser.add_argument('-s', '--seed', type=int, help="Seed value")
    parser.add_argument('-w', '--workers', type=int, default=4, help="Number of workers")
    parser.add_argument('-e', '--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('-dr', '--dropout', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('-v', '--val_split', type=float, default=0.0, help="Validation split")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help="Learning rate")
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0001, help="Weight decay")
    parser.add_argument('-mom', '--momentum', type=float, default=0.9, help="Momentum")
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('-au', '--augment', action='store_true', help='Data augmentation')
    parser.add_argument('-lrd', '--decay', type=float, default=0.1, help="Learning rate decay")
    parser.add_argument('-de', '--decay_epoch', type=int, nargs='*', help='Learning rate decay epochs', default=[60,90])
    parser.add_argument('-li', '--layer_init', type=str, help="Layer init mode", default='he_normal')
    parser.add_argument('-bi', '--b_init', type=str, help="B init mode", default='normal')
    parser.add_argument('-bm', '--b_mean_zero', action='store_false', help="Set mean of B to 0")
    parser.add_argument('-bstd', '--bstd', type=float, default=0.05, help="B standard deviation")
    parser.add_argument('-n', '--normalize', action='store_true', help='Normalize data')
    parser.add_argument('-wmlr', '--wm_learning_rate', type=float, default=0.1, help='Weight mirroring learning rate')
    parser.add_argument('-wmwd', '--wm_weight_decay', type=float, default=0.5, help='Weight mirroring weight decay')
    parser.add_argument('-prm', '--pre_mirror', type=int, default=0, help="Number of epochs of pre mirroring")
    parser.add_argument('-mir', '--mirror', type=int, default=200, help="How often to perform weight mirroring")
    parser.add_argument('-md', '--mode', type=str, default="modulated", help="Modulated pass mode")
    parser.add_argument('-na', '--normalize_activations', action='store_true', help='Normalize activations')


    args = parser.parse_args()

    if args.config_file is not None:
        hparams = update_hparams(args.config_file)
    else:
        hparams = update_hparams_from_cfg(create_arg_cfg(args))

    logger.info(f'Input arguments: \n {args}')

    main(hparams, fast_dev_run=args.fast_dev_run, checkpoint=args.checkpoint)
