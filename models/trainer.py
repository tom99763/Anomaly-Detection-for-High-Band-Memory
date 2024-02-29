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
        self.optmal_f1 = OptimalF1(2)

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
        auroc = self.auroc(batch_preds.softmax(dim=1)[:, 1], y)
        aupr = self.aupr(batch_preds.softmax(dim=1)[:, 1], y)
        opt_f1 = self.optmal_f1(batch_preds.softmax(dim=1), y)
        self.log_dict({'val_loss': loss.item(),
                       'val_auroc': auroc,
                       'val_aupr': aupr,
                       'val_opt_f1': opt_f1
                       },
                      on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'auroc': auroc, 'aupr': aupr, 'opt_f1': opt_f1}

    def on_validation_epoch_end(self):
        auroc = self.auroc.compute()
        aupr = self.aupr.compute()
        opt_f1 = self.optmal_f1.compute()
        self.log_dict({'val_auroc': auroc, 'val_aupr': aupr, 'val_opt_f1': opt_f1})
        self.auroc.reset()
        self.aupr.reset()
        self.optmal_f1.reset()

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
        auroc = self.auroc(batch_preds.softmax(dim=1)[:, 1], y)
        aupr = self.aupr(batch_preds.softmax(dim=1)[:, 1], y)
        opt_f1 = self.optmal_f1(batch_preds.softmax(dim=1), y)
        self.log_dict({'test_loss': loss.item(),
                       'test_auroc': auroc,
                       'test_aupr': aupr,
                       'test_opt_f1': opt_f1
                       },
                      on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'auroc': auroc, 'aupr': aupr, 'opt_f1': opt_f1}

    def on_test_end(self):
        auroc = self.auroc.compute()
        aupr = self.aupr.compute()
        opt_f1 = self.optmal_f1.compute()
        self.log_dict({'test_auroc': auroc, 'test_aupr': aupr, 'test_opt_f1': opt_f1})
        self.auroc.reset()
        self.aupr.reset()
        self.optmal_f1.reset()

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

