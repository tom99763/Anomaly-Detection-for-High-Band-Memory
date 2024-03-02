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
        self.aupr = AUPR()

    def training_step(self, batch, batch_idx):
        self.train()
        x, y = batch
        batch_preds, batch_regions = self.model(x)
        batch_preds = torch.stack(batch_preds, dim=0)
        if self.level == 'node':
            loss = self.loss(batch_preds, batch_regions)
        else:
            loss = self.loss(batch_preds, y.long())
        self.log('loss:', loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch):
        self.eval()
        x, y = batch
        batch_preds, batch_regions = self.model(x)
        batch_preds = torch.stack(batch_preds, dim=0)
        if self.level == 'node':
            loss = self.loss(batch_preds, batch_regions)
        else:
            loss = self.loss(batch_preds, y.long())
        #metrics
        self.auroc.update(batch_preds.softmax(dim=1)[:, 1], y)
        self.aupr.update(batch_preds.softmax(dim=1)[:, 1], y)
        self.log_dict({'val_loss': loss.item(),
                       'val_auroc': self.auroc.compute(),
                       'val_aupr': self.aupr.compute(),
                       },
                      on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'auroc': self.auroc.compute(), 'aupr': self.aupr.compute()}

    def on_validation_epoch_end(self):
        auroc = self.auroc.compute()
        aupr = self.aupr.compute()
        self.log_dict({'val_auroc': auroc, 'val_aupr': aupr})
        self.auroc.reset()
        self.aupr.reset()

    def test_step(self, batch):
        self.eval()
        x, y = batch
        batch_preds, batch_regions = self.model(x)
        batch_preds = torch.stack(batch_preds, dim=0)
        if self.level == 'node':
            loss = self.loss(batch_preds, batch_regions)
        else:
            loss = self.loss(batch_preds, y.long())
        # metrics
        self.auroc.update(batch_preds.softmax(dim=1)[:, 1], y)
        self.aupr.update(batch_preds.softmax(dim=1)[:, 1], y)
        self.log_dict({'test_loss': loss.item(),
                       'test_auroc': self.auroc.compute(),
                       'test_aupr': self.aupr.compute(),
                       },
                      on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'auroc': self.auroc.compute(), 'aupr': self.aupr.compute()}

    def on_test_end(self):
        auroc = self.auroc.compute()
        aupr = self.aupr.compute()
        #self.log_dict({'test_auroc': auroc, 'test_aupr': aupr})
        self.auroc.reset()
        self.aupr.reset()

    def backward(self, loss):
        loss.backward(retain_graph=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=1e-3
        )
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, 0.8, last_epoch=-1, verbose=True)
        return {"optimizer": optimizer,"lr_scheduler": scheduler}


    @property
    def trainer_arguments(self):
        """Set model-specific trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

