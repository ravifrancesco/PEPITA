from audioop import avg
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

import torchmetrics

import pytorch_lightning as pl

from loguru import logger

from pepita.models.FCnet import FCNet

from ..dataset import datapool, get_data_info
from . import config

class PEPITATrainer(pl.LightningModule):
    r"""Class for training models using the PEPITA algorithm
    """

    def __init__(self, hparams):
        super(PEPITATrainer, self).__init__()
        self.automatic_optimization = False # to stop pytorch_lightning from using the optimizer

        self.hparams.update(hparams)

        # Loading datasets
        self.train_dataloader_v, self.val_dataloader_v, self.test_dataloader_v = datapool(
                self.hparams.DATASET,
                self.hparams.TRAINING.BATCH_SIZE,
                val_split=self.hparams.TRAINING.VAL_SPLIT,
                augment=self.hparams.TRAINING.AUGMENT, 
                num_workers=self.hparams.HARDWARE.NUM_WORKERS
            )
        img_w, n_ch, self.n_classes = get_data_info(self.hparams.DATASET)

        # Loading model
        if self.hparams.MODEL_ARCH == 'fcnet':
            from ..models import FCnet
            self.input_size = img_w * img_w * n_ch
            hidden_layers = self.hparams.MODEL.FCNet.HIDDEN_LAYER_SIZES
            self.output_size = self.n_classes
            layers = [self.input_size] + hidden_layers + [self.output_size]
            self.model = FCNet(
                layers,
                B_mean_zero=self.hparams.PEPITA.B_MEAN_ZERO, 
                Bstd=self.hparams.PEPITA.BSTD,
                p=self.hparams.TRAINING.DROPOUT_P
            )
            self.reshape = True
            self.example_input_array = torch.rand((1, self.input_size))
        else:
            logger.error(f'Model \'{self.hparams.MODEL}\' is undefined!')
            exit()

        # Setting hyperparams
        self.lr = self.hparams.TRAINING.LR
        self.bs = self.hparams.TRAINING.BATCH_SIZE
        self.lr_decay = self.hparams.TRAINING.LR_DECAY
        self.decay_epoch = self.hparams.TRAINING.DECAY_EPOCH

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            imgs, gt = batch
            if self.reshape:
                imgs = imgs.reshape(-1, self.input_size)
            outputs = self(imgs)
        
            one_hot = F.one_hot(gt, num_classes=self.n_classes)

            # Update model using the PEPITA learning rule
            self.model.update(imgs, outputs-one_hot, self.lr, self.bs)

            loss = F.cross_entropy(outputs, gt)
            self.train_acc(torch.argmax(outputs, -1), gt)

        return {'train_loss': loss, 'train_acc': self.train_acc}

    def training_epoch_end(self, outputs):
        # Scales LR according to the learning rate decay rule
        if self.current_epoch in self.hparams.TRAINING.DECAY_EPOCH:
            logger.info(f'Epoch {self.current_epoch} - Learning rate decay: {self.lr} -> {self.lr*self.lr_decay}')
            self.lr = self.lr*self.lr_decay

        avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss, 'train_acc': self.train_acc, 'step': self.current_epoch}
        self.log_dict(tensorboard_logs, prog_bar=True, on_step=False, on_epoch=True)
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            imgs, gt = batch
            if self.reshape:
                imgs = imgs.reshape(-1, self.input_size)
            outputs = self(imgs)
        
            val_loss = F.cross_entropy(outputs, gt)   
            self.val_acc(torch.argmax(outputs, -1), gt)

        return {'val_loss': val_loss, 'val_acc': self.val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': self.val_acc, 'step': self.current_epoch}
        self.log_dict(tensorboard_logs, prog_bar=True, on_step=False, on_epoch=True)


    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            imgs, gt = batch
            if self.reshape:
                imgs = imgs.reshape(-1, self.input_size)
            outputs = self(imgs)
        
            test_loss = F.cross_entropy(outputs, gt)   
            tensorboard_logs = {'test_loss': test_loss}
            self.log_dict(tensorboard_logs)
            self.log("test_loss", test_loss)
            logger.info(f'Testing loss: {test_loss}')
            logger.info(f'Testing acc: {self.test_acc}')

    def configure_optimizers(self):
        return None

    def train_dataloader(self):
        return self.train_dataloader_v
    
    def val_dataloader(self):
        return self.val_dataloader_v

    def test_dataloader(self):
        return self.test_dataloader_v
