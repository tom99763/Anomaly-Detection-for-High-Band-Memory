import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
from .model import *

class RegionClip(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = RegionClipModel(config)
        self.level = self.model.level
        self.loss = prior_cross_entropy
    def training_step(self, batch):
        batch_preds, batch_regions = self.model(batch['image'])
        loss = self.loss(batch_preds, batch_regions)
        self.log('loss:', loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch):
        pass

    def configure_optimizers(self):
        return optim.Adam(
            params=self.model.parameters(),
            lr=0.001
        )

    @property
    def trainer_arguments(self):
        """Set model-specific trainer arguments."""
        return {}
