import numpy as np

from scipy import spatial

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import torchmetrics

import pytorch_lightning as pl

from loguru import logger

from pepita.models import modelpool
from ..dataset import datapool, get_data_info
from . import config


class PEPITATrainer(pl.LightningModule):
    r"""Class for training models using the PEPITA algorithm"""

    def __init__(self, hparams):
        super(PEPITATrainer, self).__init__()
        self.automatic_optimization = (
            False  # to stop pytorch_lightning from using the optimizer
        )

        self.hparams.update(hparams)

        # Loading datasets
        (
            self.train_dataloader_v,
            self.val_dataloader_v,
            self.test_dataloader_v,
        ) = datapool(
            self.hparams.DATASET,
            self.hparams.TRAINING.BATCH_SIZE,
            val_split=self.hparams.TRAINING.VAL_SPLIT,
            augment=self.hparams.TRAINING.AUGMENT,
            num_workers=self.hparams.HARDWARE.NUM_WORKERS,
            normalize=self.hparams.TRAINING.NORMALIZE,
        )

        # Loading model
        self.model, self.input_size, self.n_classes, self.reshape = modelpool(
            self.hparams.MODEL_ARCH, self.hparams
        )
        self.example_input_array = torch.rand((1, self.input_size))

        # Setting hyperparams
        self.lr = self.hparams.TRAINING.LR
        self.mom = self.hparams.TRAINING.MOMENTUM
        self.wd = hparams.TRAINING.WD
        self.bs = self.hparams.TRAINING.BATCH_SIZE
        self.lr_decay = self.hparams.TRAINING.LR_DECAY
        self.decay_epoch = self.hparams.TRAINING.DECAY_EPOCH

        self.wmlr = self.hparams.TRAINING.WMLR
        self.wmwd = self.hparams.TRAINING.WMWD
        self.premirror = self.hparams.TRAINING.PRE_MIRROR
        self.mirror = self.hparams.TRAINING.MIRROR

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.wmlr = 0.01
        self.wmwd = 0.0001

        #TEST
        self.save = hparams.MODEL_DIR

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            imgs, gt = batch
            if self.reshape:
                imgs = imgs.reshape(-1, self.input_size)
            outputs = self(imgs)

            one_hot = F.one_hot(gt, num_classes=self.n_classes)

            # Compute modulated activations
            if self.current_epoch >= self.premirror:
                self.model.modulated_forward(imgs, outputs - one_hot, imgs.shape[0])

            # Perform weight mirroring
            if (
                self.current_epoch < self.premirror
                or not (self.current_epoch + 1) % self.mirror
            ):
                self.model.mirror_weights(imgs.shape[0])

            loss = F.cross_entropy(outputs, gt)
            self.train_acc(torch.argmax(outputs, -1), gt)
            
            if self.current_epoch >= 5:
                opt = self.optimizers()
                opt.step()
                opt.zero_grad()

            opt_w, opt_b = self.optimizers()
            opt_w.step()
            opt_w.zero_grad()
            opt_b.step()
            opt_b.zero_grad()

        return {"train_loss": loss, "train_acc": self.train_acc}

    def training_epoch_end(self, outputs):

        if self.current_epoch in self.hparams.TRAINING.DECAY_EPOCH:
            logger.info(
                f"Epoch {self.current_epoch} - Learning rate decay: {self.lr} -> {self.lr*self.lr_decay}"
            )
            self.lr = self.lr * self.lr_decay
            opt_w, opt_b = self.optimizers()
            opt_w.param_groups[0]["lr"] = self.lr

        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        tensorboard_logs = {
            "train_loss": avg_loss,
            "train_acc": self.train_acc,
            "angle": self.model.compute_angle(),
            "weight_norms": self.model.get_weights_norm(),
            "step": self.current_epoch,
        }
        self.log_dict(tensorboard_logs, prog_bar=True, on_step=False, on_epoch=True)

        if self.current_epoch == 46:
            self.save_checkpoint(f'{self.save}/model_46.ckpt')

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            imgs, gt = batch
            if self.reshape:
                imgs = imgs.reshape(-1, self.input_size)
            outputs = self(imgs)

            val_loss = F.cross_entropy(outputs, gt)
            self.val_acc(torch.argmax(outputs, -1), gt)

        return {"val_loss": val_loss, "val_acc": self.val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_acc": self.val_acc,
            "step": self.current_epoch,
        }
        self.log_dict(tensorboard_logs, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            imgs, gt = batch
            if self.reshape:
                imgs = imgs.reshape(-1, self.input_size)
            outputs = self(imgs)

            test_loss = F.cross_entropy(outputs, gt)
            n_correct_pred = torch.sum(torch.argmax(outputs, -1) == gt).item()

        return {
            "test_loss": test_loss,
            "n_correct_pred": n_correct_pred,
            "n_pred": len(gt),
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(
            x["n_pred"] for x in outputs
        )
        logger.info(f"Testing loss: {avg_loss}")
        logger.info(f"Testing acc: {test_acc}")

    def configure_optimizers(self):

        # optimizer for weights
        opt_w = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=self.mom, weight_decay=self.wd
        )
        # optimizer for feedback matrices
        opt_b = torch.optim.SGD(self.model.Bs, lr=self.wmlr, weight_decay=self.wmwd)
        return opt_w, opt_b

    def train_dataloader(self):
        return self.train_dataloader_v

    def val_dataloader(self):
        # return self.val_dataloader_v FIXME change
        return self.test_dataloader_v

    def test_dataloader(self):
        return self.test_dataloader_v

    @torch.no_grad()
    def compute_angle(self):
        r"""Returns angle between feedforward matrix and feedback matrix TODO adapt other functions to support multiple angles
        """
        return self.model.compute_angle()

