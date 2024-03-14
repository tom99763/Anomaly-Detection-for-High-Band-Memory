import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
from .model_ct import *
from train_tools.metrics import *
from .losses import *

class RegionClipCT(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = RegionClipCTModel(config)
        self._transform = self.model._transform
        self.config = config
        self.pad_green = config['pad']['pad_green']
        self.use_margin = config['loss']['use_margin']
        self.beta = config['loss']['beta']

        #metrics
        self.auroc = AUROC()
        self.aupr = AUPR()
        self.step = 0.

    def training_step(self, batch, batch_idx):
        self.train()
        x, y = batch
        if not self.step%2:
            batch_region_node_preds, batch_region_nodes, \
            batch_region_node_preds_aug, batch_region_nodes_aug,\
                batch_unlabeled_idx = self.model(x, self.pad_green, True, True)
            l_c, l_div = div_consistency_loss(
                batch_region_nodes,
                batch_region_nodes_aug,
                batch_region_node_preds,
                batch_region_node_preds_aug,
                batch_unlabeled_idx
            )
            loss = self.beta * l_c + l_div
            l_ce = torch.tensor(0.)
        else:
            batch_region_node_preds, batch_region_nodes, \
            batch_region_node_preds_aug, batch_region_nodes_aug, \
            batch_unlabeled_idx = self.model(x, self.pad_green, True, False)
            l_c, l_div = div_consistency_loss(
                batch_region_nodes,
                batch_region_nodes_aug,
                batch_region_node_preds,
                batch_region_node_preds_aug,
                batch_unlabeled_idx
            )
            l_ce = cross_entropy_loss(batch_region_node_preds, batch_unlabeled_idx)
            loss = l_ce + self.beta * l_c

        self.step+=1
        self.log_dict({'loss:': loss.item(),
                       'l_ce': l_ce.item(),
                       'l_c':l_c.item(),
                       'l_div': l_div.item()},
                      on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch):
        self.eval()
        x, y = batch
        batch_region_node_preds, batch_region_nodes, \
        batch_region_node_preds_aug, batch_region_nodes_aug, \
        batch_unlabeled_idx = self.model(x, self.pad_green)

        batch_preds = []
        for i in range(x.shape[0]):
            region_node_preds = batch_region_node_preds[i].softmax(dim=-1)
            region_node_preds = region_node_preds[:, 1].max()[None,]
            batch_preds.append(region_node_preds)
        batch_preds = torch.cat(batch_preds, dim=0)  # (N, )

        #metrics
        self.auroc.update(batch_preds, y)
        self.aupr.update(batch_preds, y)

        #auroc = self.auroc.compute()
        #aupr = self.aupr.compute()

        self.log_dict({'val_auroc': self.auroc.compute(),
                       'val_aupr': self.aupr.compute(),
                       },
                      on_epoch=True, prog_bar=True, logger=True)
        return {'auroc': self.auroc.compute(), 'aupr': self.aupr.compute()}

    def on_validation_epoch_end(self):
        auroc = self.auroc.compute()
        aupr = self.aupr.compute()
        self.log_dict({'val_auroc': auroc, 'val_aupr': aupr})
        self.auroc.reset()
        self.aupr.reset()

    def test_step(self, batch):
        self.eval()
        x, y = batch
        batch_region_node_preds, batch_region_nodes, \
        batch_region_node_preds_aug, batch_region_nodes_aug, \
        batch_unlabeled_idx = self.model(x, self.pad_green)

        batch_preds = []
        for i in range(x.shape[0]):
            region_node_preds = batch_region_node_preds[i].softmax(dim=-1)
            region_node_preds = region_node_preds[:, 1].max()[None,]
            batch_preds.append(region_node_preds)
        batch_preds = torch.cat(batch_preds, dim=0)  # (N, )

        # metrics
        self.auroc.update(batch_preds, y)
        self.aupr.update(batch_preds, y)

        # auroc = self.auroc.compute()
        # aupr = self.aupr.compute()

        self.log_dict({'test_auroc': self.auroc.compute(),
                       'test_aupr': self.aupr.compute(),
                       },
                      on_epoch=True, prog_bar=True, logger=True)
        return {'auroc': self.auroc.compute(), 'aupr': self.aupr.compute()}

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
