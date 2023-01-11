import os
import resource
import sys
from threading import local
import torch
import argparse
import math

from loguru import logger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import ray
from ray import tune, air
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.result import NODE_IP

sys.path.append('')

from pepita.core.trainer import PEPITATrainer
from pepita.core.config import update_hparams_from_dict
from pepita.utils.train_utils import seed_everything
from utils import create_grid_search_dict

def main(cfg_dict, n_cpus=4, n_gpus=1, resume=False):

    log_dir = f"experiments/{cfg_dict['EXP_NAME']}/logs"
    seed_everything(cfg_dict["SEED_VALUE"])

    logger.add(
        os.path.join(log_dir, 'grid_search.log'),
        level='INFO',
        colorize=False,
    )

    asha_scheduler = ASHAScheduler(
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1
    )

    class ClearNodeIpCallback(tune.Callback):
        def __init__(self):
         self.reset_trials = set()

        def on_trial_result(self, iteration, trials, trial, result):
            if trial.trial_id not in self.reset_trials:
                trial.last_result.pop(NODE_IP, None)
                self.reset_trials.add(trial.trial_id)

    logger.info('*** Started Grid Search ***')

    results = tune.run(
        tune.with_parameters(
            train_model,
            n_gpus=min(1, n_gpus)
        ),
        resources_per_trial={"cpu" : min(4, n_cpus), "gpu": min(1, n_gpus)},
        metric="val_acc",
        mode="max",
        num_samples=1,
        config=cfg_dict,
        # scheduler=asha_scheduler,
        name=cfg_dict['EXP_NAME'],
        local_dir=f"experiments",
        max_failures=5,
        resume=resume,
        callbacks=[ClearNodeIpCallback()],
    )

    logger.info('*** Grid Search Ended ***')

    logger.info(f'Best result: {str(results.best_result)}')
    logger.info(f'Best config: \n {str(results.best_config)}')

    logger.info("Results")
    logger.info('\t'+ results.dataframe().to_string().replace('\n', '\n\t'))
    results.dataframe().to_pickle(f"experiments/{cfg_dict['EXP_NAME']}/trials.pkl")

def train_model(config, fast_dev_run=False, n_gpus=0):

    hparams = update_hparams_from_dict(str(config))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PEPITATrainer(hparams=hparams)

    metrics = {"val_acc": "val_acc"}

    trainer = pl.Trainer(
        #cgpus=1,
        max_epochs=hparams.TRAINING.MAX_EPOCHS,
        logger=False,
        checkpoint_callback=False,
        #enable_checkpointing=ckpt_callback,
        #checkpoint_callback=ckpt_callback,
        callbacks=[TuneReportCallback(metrics, on="validation_end")],
        #terminate_on_nan=True,
        progress_bar_refresh_rate=0,
        check_val_every_n_epoch=hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH,
        #resume_from_checkpoint=hparams.TRAINING.RESUME,
        num_sanity_val_steps=0,
        fast_dev_run=fast_dev_run,
        gpus=math.ceil(n_gpus),
        #limit_val_batches=0 if not hparams.TRAINING.VAL_SPLIT else 1 FIXME change
    )

    trainer.fit(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-en', '--exp_name', type=str, default='exp', help="Experiment name")
    parser.add_argument('-res', '--resume', action='store_true', help='Resume experiment')
    parser.add_argument('-cpu', '--cpus', type=int, default=4, help="Number of CPU cores")
    parser.add_argument('-gpu', '--gpus', type=int, default=0, help="Number of GPUs")
    parser.add_argument('-a', '--arch', type=str, default='fcnet', help="Model architecture")
    parser.add_argument('-ls', '--layer_sizes', nargs='*', type=int, default=[1024], help="sizes of the layers")
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', help="Dataset")
    parser.add_argument('-s', '--seed', type=int, help="Seed value")
    parser.add_argument('-w', '--workers', type=int, default=4, help="Number of workers")
    parser.add_argument('-e', '--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('-dr', '--dropout', nargs='*', type=float, default=[0.1], help="Dropout rate")
    parser.add_argument('-v', '--val_split', type=float, default=0.0, help="Validation split")
    parser.add_argument('-lr', '--learning_rate', nargs='*', type=float, default=[0.01], help="Learning rate")
    parser.add_argument('-wd', '--weight_decay', nargs='*', type=float, default=[0.0001], help="Weight decay")
    parser.add_argument('-mom', '--momentum', nargs='*', type=float, default=[0.9], help="Learning rate")
    parser.add_argument('-bs', '--batch_size', nargs='*', type=int, default=[64], help="Batch size")
    parser.add_argument('-au', '--augment', action='store_true', help='Data augmentation')
    parser.add_argument('-lrd', '--decay', nargs='*', type=float, default=[0.1], help="Learning rate decay")
    parser.add_argument('-de', '--decay_epoch', type=int, nargs='+', action='append', help='Learning rate decay epochs', default=[[60,90]])
    parser.add_argument('-li', '--layer_init', type=str, help="Layer init mode", default='he_normal')
    parser.add_argument('-bi', '--b_init', type=str, help="B init mode", default='normal')
    parser.add_argument('-bm', '--b_mean_zero', action='store_false', help="Mean of B is 0")
    parser.add_argument('-bstd', '--bstd', nargs='*', type=float, default=[0.05], help="B standar deviation")
    parser.add_argument('-n', '--normalize', action='store_true', help='normalize data')
    parser.add_argument('-wmlr', '--wm_learning_rate', nargs='*', type=float, default=[0.01], help='Weight mirroring learning rate')
    parser.add_argument('-wmwd', '--wm_weight_decay', nargs='*', type=float, default=[0.0001], help='Weight mirroring weight decay')
    parser.add_argument('-prm', '--pre_mirror', nargs='*', type=int, default=[0], help="Number of epochs of pre mirroring")
    parser.add_argument('-mir', '--mirror', nargs='*', type=int, default=[200], help="How often to perform weight mirroring")
    parser.add_argument('-md', '--mode', type=str, default="modulated", help="Modulated pass mode")
    parser.add_argument('-na', '--normalize_activations', action='store_true', help='normalize activations')

    args = parser.parse_args()

    cfg_dict = create_grid_search_dict(args)

    ray.init(address="local", num_cpus=args.cpus, num_gpus=args.gpus)

    main(cfg_dict, n_cpus=args.cpus, n_gpus=args.gpus, resume=args.resume)

    ray.shutdown()