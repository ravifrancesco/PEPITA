import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

import pytorch_lightning as pl

from loguru import logger

from pepita.models.FCnet import FCNet

from ..dataset import datapool, get_data_info
from . import config

class PEPITATrainer(pl.LightningModule):

    def __init__(self, dataset, hparams):
        super(PEPITATrainer, self).__init__()
        self.automatic_optimization = False

        self.hparams.update(hparams)

        # Loading datasets
        self.train_dataloader, self.val_dataloader, self.test_dataloader = datapool(
                self.hparams.TRAINING.BATCH_SIZE,
                val_split=self.hparams.TRAINING.VAL_SPLIT,
                augment=self.hparams.TRAINING.AUGMENT, 
                num_workers=self.hparams.HARDWARE.NUM_WORKERS
            )
        img_w, n_ch, self.n_classes = get_data_info(dataset)

        # Loading model
        if self.hparams.MODEL == 'fcnet':
            from ..models import FCnet
            self.input_size = img_w * img_w * n_ch
            hidden_layers = self.hparams.MODEL.FCNet.HIDDEN_LAYER_SIZES
            self.output_size = n_classes
            layers = [self.input_size] + hidden_layers + [self.output_size]
            self.model = FCNet(
                layers,
                B_mean_zero=self.hparams.PEPITA.B_MEAN_ZERO,
                Bstd=self.hparams.PEPITA.BSTD
            )
            self.reshape = True
        else:
            logger.error(f'{self.hparams.MODEL} is undefined!')
            exit()

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.OPTIMIZER.LR,
            weight_decay=self.hparams.OPTIMIZER.WD
        )

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            imgs, gt = batch
            if self.reshape:
                imgs = imgs.reshape(-1, self.input_size)
            outputs = self(imgs)
        
            one_hot = F.one_hot(gt, num_classes=self.n_classes)
            self.model.update(imgs, outputs-one_hot)

            loss = F.cross_entropy(outputs, gt)   
            tensorboard_logs = {'training_loss': loss}
            self.log_dict(tensorboard_logs)

        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            imgs, gt = batch
            if self.reshape:
                imgs = imgs.reshape(-1, self.input_size)
            outputs = self(imgs)
        
            val_loss = F.cross_entropy(outputs, gt)   
            tensorboard_logs = {'val_loss': val_loss}
            self.log_dict(tensorboard_logs)
            self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            imgs, gt = batch
            if self.reshape:
                imgs = imgs.reshape(-1, self.input_size)
            outputs = self(imgs)
        
            val_loss = F.cross_entropy(outputs, gt)   
            tensorboard_logs = {'test_loss': val_loss}
            self.log_dict(tensorboard_logs)
            self.log("test_loss", val_loss)

    def train_dataloader(self):
        return self.train_dataloader
    
    def val_dataloader(self):
        return self.val_dataloader
    
    def test_dataloader(self):
        return self.test_dataloader 
