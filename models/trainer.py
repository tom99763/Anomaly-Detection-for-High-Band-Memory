import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
from .model import *
from train_tools.metrics import *

class RegionClip(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = RegionClipModel(config)
        self._transform = self.model._transform
        self.level = self.model.level
        self.loss = prior_cross_entropy if self.level == 'node' else nn.CrossEntropyLoss()
        self.config = config

        #metrics
        self.auroc = AUROC()

    def training_step(self, batch, batch_idx):
        self.train()
        x, y = batch
        batch_preds, batch_regions = self.model(x)
        if self.level == 'node':
            loss = self.loss(batch_preds, batch_regions)
        else:
            batch_preds = torch.stack(batch_preds, dim=0)
            loss = self.loss(batch_preds, y.long())
        self.log('loss:', loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch):
        self.eval()
        x, y = batch
        batch_preds, batch_regions = self.model(x)
        if self.level == 'node':
            loss = self.loss(batch_preds, batch_regions)
        else:
            batch_preds = torch.stack(batch_preds, dim=0)
            loss = self.loss(batch_preds, y.long())
        auroc = self.auroc(batch_preds[:, 1], y)
        self.log_dict({'val_loss': loss.item(), 'val_auroc': auroc},
                 on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'auroc': auroc}

    def test_step(self, batch):
        self.eval()
        x, y = batch
        batch_preds, batch_regions = self.model(x)
        if self.level == 'node':
            loss = self.loss(batch_preds, batch_regions)
        else:
            batch_preds = torch.stack(batch_preds, dim=0)
            loss = self.loss(batch_preds, y.long())
        auroc = self.auroc(batch_preds[:, 1], y)
        self.log_dict({'val_loss': loss.item(), 'val_auroc': auroc},
                      on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'auroc': auroc}

    def backward(self, loss):
        loss.backward(retain_graph=True)

    def configure_optimizers(self):
        #optim.SGD(params=self.model.parameters(), lr=0.001)
        return optim.Adam(
            params=self.model.parameters(),
            lr=0.001
        )

    @property
    def trainer_arguments(self):
        """Set model-specific trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

