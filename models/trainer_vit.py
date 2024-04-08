import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
from .model_vit import *
from train_tools.metrics import *
from .losses import *
from torchmetrics.classification import BinaryF1Score
import torch.nn.functional as F
from .threshold import *

class RegionClipViT(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = RegionClipViTModel(config)
        self.config = config
        self.mode = self.config['st']['mode']
        self.pad_green = self.config['pad']['pad_green']

        #metrics
        self.auroc = AUROC().cuda()
        self.aupr = AUPR().cuda()
        self.f1_score = BinaryF1Score().cuda()
        self.f1_adapt = F1AdaptiveThreshold()
        self.min_max = MinMax()
        self.step = 0.
        self.thr = self.f1_adapt.value
        self.min, self.max = self.min_max.min, self.min_max.max
        self.beta = config['st']['beta']
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.cuda()
        if self.mode == 'teacher':
            self.model.teacher.train()
            img_preds, node_preds_t, nodes_t, nodes_s, anomap = self.model(x)
            loss = nn.CrossEntropyLoss()(node_preds_t, img_preds.softmax(dim=-1))+ \
                   self.beta *(torch.sum(img_preds**2, dim=-1).mean() +
                   torch.sum(node_preds_t**2, dim=-1).mean())

        elif self.mode == 'student':
            self.model.teacher.eval()
            self.model.student.train()
            img_preds, node_preds_t, nodes_t, nodes_s, anomap = self.model(x)
            #loss = nn.MSELoss(reduction='none')(nodes_s, nodes_t) #(b, N, 640)
            #loss = loss.reshape(-1, 225 * 640)
            #q = torch.quantile(loss, 0.8, dim=-1, keepdim=True) #(b, 1)
            #loss = loss[loss > q].sum()
            loss = vitst_loss(nodes_t, nodes_s)

        self.step += 1
        self.log_dict({'loss:': loss.item(),
                       },
                      on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch):
        x, y = batch
        y = y.cuda()

        if self.mode == 'student':
            self.model.teacher.eval()
            self.model.student.eval()
            img_preds, node_preds_t, nodes_t, nodes_s, anomap = self.model(x)

            self.min_max.update(anomap)
            scores, _ = anomap.flatten(1, 3).max(dim=-1)

            # update score
            self.f1_adapt.update(scores, y)
            scores = normalize(scores, self.min, self.max, self.thr)


            # metrics
            self.auroc.update(scores, y)
            self.aupr.update(scores, y)
            self.f1_score.update(scores, y)

            self.log_dict({'val_f1': self.f1_score.compute(),
                           },
                          on_epoch=True, prog_bar=True, logger=True)

        elif self.mode == 'teacher':
            self.model.teacher.eval()
            img_preds, node_preds_t, nodes_t, nodes_s, anomap = self.model(x)
            loss = nn.CrossEntropyLoss()(node_preds_t, img_preds.softmax(dim=-1)) + \
                   self.beta * (torch.sum(img_preds ** 2, dim=-1).mean() +
                                torch.sum(node_preds_t ** 2, dim=-1).mean())

            print('val', y)
            print('val', img_preds.softmax(dim=-1)[:, 1])
            print('val', node_preds_t.softmax(dim=-1)[:, 1])

            self.log_dict({'val_loss': loss.item(),
                           },
                          on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        if self.mode == 'student':
            auroc = self.auroc.compute()
            aupr = self.aupr.compute()
            f1_score = self.f1_score.compute()
            self.log_dict(
                {'val_auroc': auroc, 'val_aupr': aupr, 'val_f1_score': f1_score,
                 'val_min': self.min, 'val_max': self.max, 'f1_thr': self.thr
                 })
            self.auroc.reset()
            self.aupr.reset()
            self.f1_score.reset()
            self.thr = self.f1_adapt.compute()
            self.min, self.max = self.min_max.compute()
            self.f1_adapt.reset()
            self.min_max.reset()

    def test_step(self, batch):
        x, y = batch
        y = y.cuda()

        if self.mode == 'student':
            self.model.teacher.eval()
            self.model.student.eval()
            img_preds, node_preds_t, nodes_t, nodes_s, anomap = self.model(x)

            self.min_max.update(anomap)
            scores, _ = anomap.flatten(1, 3).max(dim=-1)

            # update score
            self.f1_adapt.update(scores, y)
            scores = normalize(scores, self.min, self.max, self.thr)

            # metrics
            self.auroc.update(scores, y)
            self.aupr.update(scores, y)
            self.f1_score.update(scores, y)

    def on_test_end(self):
        if self.mode == 'student':
            auroc = self.auroc.compute()
            aupr = self.aupr.compute()
            f1_score = self.f1_score.compute()
            self.logger.log_metrics({'auroc': auroc, 'aupr': aupr, 'f1_score': f1_score})
            self.auroc.reset()
            self.aupr.reset()
            self.f1_score.reset()

    def backward(self, loss):
        loss.backward()

    def configure_optimizers(self):
        optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=1e-3
        )
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, 0.95, last_epoch=-1, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint):
        checkpoint['thr'] = self.thr
        checkpoint['min'] = self.min
        checkpoint['max'] = self.max

    def on_load_checkpoint(self, checkpoint):
        self.thr = checkpoint['thr']
        self.min = checkpoint['min']
        self.max = checkpoint['max']

    @property
    def trainer_arguments(self):
        """Set model-specific trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

