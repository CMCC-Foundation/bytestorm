# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ByteStorm Library for Tropical Cyclone Tracking           #
# Copyright (c) 2025 CMCC Foundation                        #
# Licensed under The MIT License [see LICENSE for details]  #
# Written by Davide Donno                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from lightning.pytorch.callbacks import Callback
from lightning import Trainer, LightningModule
import pandas as pd
import os


class BenchmarkCSV(Callback):
    def __init__(self, filename) -> None:
        self.filename = filename

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_rank == 0:
            # get the metrics from the trainer
            metrics = pl_module.callback_metrics
            # create csv file if not exists
            if not os.path.exists(self.filename):
                # define csv columns
                columns = [key for key in metrics.keys()]
                # create empty DataFrame
                self.df = pd.DataFrame(columns=columns)
                # save the DataFrame to disk
                self.df.to_csv(self.filename)
            else:
                # get the DataFrame from disk
                self.df = pd.read_csv(self.filename, index_col=0)
                # get the columns
                columns = self.df.columns
            # create the row to be added to DataFrame
            row = [metrics[col].item() for col in columns]
            # add to the DataFrame the row
            self.df.loc[len(self.df.index)] = row
            # store the data to the csv file
            self.df.to_csv(self.filename)
        return
