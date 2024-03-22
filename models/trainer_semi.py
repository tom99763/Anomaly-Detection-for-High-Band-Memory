import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
from .model_semi import *
from train_tools.metrics import *
from .losses import *
from torchmetrics.classification import BinaryF1Score
import torch.nn.functional as F

class RegionClipSemi(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = RegionClipSemiModel(config)
        self._transform = self.model._transform
        self.config = config
        self.pad_green = config['pad']['pad_green']

        #metrics
        self.auroc = AUROC().cuda()
        self.aupr = AUPR().cuda()
        self.f1_score = BinaryF1Score().cuda()
        self.step = 0.

    def training_step(self, batch, batch_idx):
        self.train()
        x, y = batch
        y = y.cuda()
        batch_region_node_preds, batch_region_nodes,\
        batch_text_embs,batch_regions = self.model(x, self.pad_green)
        loss = cross_entropy(batch_region_node_preds, y.long())

        self.step += 1
        self.log_dict({'loss:': loss.item(),
                       },
                      on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch):
        self.eval()
        x, y = batch
        y = y.cuda()
        batch_region_node_preds, batch_region_nodes, \
        batch_text_embs, batch_regions = self.model(x, self.pad_green)
        batch_preds = batch_region_node_preds.softmax(dim=-1)[:, 1]

        #metrics
        self.auroc.update(batch_preds, y)
        self.aupr.update(batch_preds, y)
        self.f1_score.update(batch_preds, y)

        self.log_dict({'val_auroc': self.auroc.compute(),
                       'val_aupr': self.aupr.compute(),
                       'val_f1_score': self.f1_score.compute()
                       },
                      on_epoch=True, prog_bar=True, logger=True)
        return {'auroc': self.auroc.compute(), 'aupr': self.aupr.compute(),
                'f1_score':self.f1_score.compute()
                }

    def on_validation_epoch_end(self):
        auroc = self.auroc.compute()
        aupr = self.aupr.compute()
        f1_score = self.f1_score.compute()
        self.log_dict({'val_auroc': auroc, 'val_aupr': aupr, 'val_f1_score':f1_score})
        self.auroc.reset()
        self.aupr.reset()
        self.f1_score.reset()

    def test_step(self, batch):
        self.eval()
        x, y = batch
        y = y.cuda()
        batch_region_node_preds, batch_region_nodes, \
        batch_text_embs, batch_regions = self.model(x, self.pad_green)
        batch_preds = batch_region_node_preds.softmax(dim=-1)[:, 1]

        # metrics
        self.auroc.update(batch_preds, y)
        self.aupr.update(batch_preds, y)
        self.f1_score.update(batch_preds, y)

        self.log_dict({'test_auroc': self.auroc.compute(),
                       'test_aupr': self.aupr.compute(),
                       'test_f1_score': self.f1_score.compute()
                       },
                      on_epoch=True, prog_bar=True, logger=True)
        return {'auroc': self.auroc.compute(), 'aupr': self.aupr.compute(),
                'f1_score': self.f1_score.compute()
                }

    def on_test_end(self):
        self.auroc.reset()
        self.aupr.reset()
        self.f1_score.reset()

    def backward(self, loss):
        loss.backward(retain_graph=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=1e-3
        )
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, 0.9, last_epoch=-1, verbose=True)
        return {"optimizer": optimizer,"lr_scheduler": scheduler}


    @property
    def trainer_arguments(self):
        """Set model-specific trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

