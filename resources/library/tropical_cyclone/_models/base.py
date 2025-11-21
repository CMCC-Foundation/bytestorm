# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ByteStorm Library for Tropical Cyclone Tracking           #
# Copyright (c) 2025 CMCC Foundation                        #
# Licensed under The GPLv3 License [see LICENSE for details]#
# Written by Davide Donno                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import lightning as L
from typing import Any


class BaseLightningModule(L.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.callback_metrics = {}

    def training_step(self, batch, batch_idx):
        # get data from the batch
        x, y = batch
        # forward pass
        y_pred = self(x)
        # compute loss
        loss = self.loss(y_pred, y)
        # define log dictionary
        log_dict = {'train_loss': loss}
        # compute metrics
        for metric in self.metrics:
            metric_value = metric(y_pred, y)
            log_dict.update({f'train_{metric.name}' : metric_value})
            self.log(f'train_{metric.name}', metric_value, prog_bar=True)
        # log the outputs
        self.callback_metrics = {**self.callback_metrics, **log_dict}
        self.log('train_loss', loss, prog_bar=True)
        # return the loss
        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        # get data from the batch
        x, y = batch
        # forward pass
        y_pred = self(x)
        # compute loss
        loss = self.loss(y_pred, y)
        # define log dictionary
        log_dict = {'val_loss': loss}
        # compute metrics
        for metric in self.metrics:
            metric_value = metric(y_pred, y)
            log_dict.update({f'val_{metric.name}' : metric_value})
            self.log(f'val_{metric.name}', metric_value, prog_bar=True)
        # log the outputs
        self.callback_metrics = {**self.callback_metrics, **log_dict}
        self.log('val_loss', loss, prog_bar=True)
        # return the loss
        return {'loss':loss}

    def configure_optimizers(self):
        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler}
