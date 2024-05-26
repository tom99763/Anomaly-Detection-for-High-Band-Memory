import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
from .model_lp import *
from train_tools.metrics import *
from .losses import *
from torchmetrics.classification import BinaryF1Score
import torch.nn.functional as F
from .threshold import *

class RegionClipLP(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = RegionClipLPModel(config)
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
        self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.cuda()
        batch_size = x.shape[0]
        if self.mode == 'teacher':
            img_preds, batch_preds_t, batch_preds_s, \
            batch_nodes_t, batch_nodes_s, batch_anomaps = self.model(x, self.pad_green)

            kl_loss = nn.KLDivLoss(reduction="batchmean")
            target = img_preds.softmax(dim=-1)
            input_ = F.log_softmax(batch_preds_t, dim=-1)
            loss = kl_loss(input_, target) + torch.sum(img_preds**2, dim=-1).mean() +\
                   torch.sum(batch_preds_t**2, dim=-1).mean()

            #loss = nn.CrossEntropyLoss()(batch_preds_t, img_preds.softmax(dim=-1))+ \
                   #torch.sum(img_preds**2, dim=-1).mean() +\
                   #torch.sum(batch_preds_t**2, dim=-1).mean()

        elif self.mode == 'student':
            img_preds, batch_preds_t, batch_preds_s, \
            batch_nodes_t, batch_nodes_s, batch_anomaps = self.model(x, self.pad_green)
            loss = torch.tensor(0.).cuda()
            for i in range(batch_size):
                node_s = batch_nodes_s[i]
                node_t = batch_nodes_t[i]
                loss_ =  nn.MSELoss(reduction='none')(node_s, node_t) #(N, d)
                q = torch.quantile(loss_, 0.8)
                loss += loss_[loss_>q].sum()

        self.step += 1
        self.log_dict({'loss:': loss.item(),
                       },
                      on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch):
        x, y = batch
        y = y.cuda()

        if self.mode == 'student':
            img_preds, batch_preds_t, batch_preds_s, \
            batch_nodes_t, batch_nodes_s, batch_anomaps = self.model(x, self.pad_green)


            scores = []
            for anomap in batch_anomaps:
                self.min_max.update(anomap)
                score, _ = anomap.max(dim=0)
                scores.append(score)
            scores = torch.cat(scores)

            # update score
            self.f1_adapt.update(scores, y)
            scores = normalize(scores, self.min, self.max, self.thr)

            # metrics
            self.auroc.update(scores, y)
            self.aupr.update(scores, y)
            self.f1_score.update(scores, y)

        elif self.mode == 'teacher':
            img_preds, batch_preds_t, batch_preds_s, \
            batch_nodes_t, batch_nodes_s, batch_anomaps = self.model(x, self.pad_green)

            kl_loss = nn.KLDivLoss(reduction="batchmean")
            target = img_preds.softmax(dim=-1)
            input_ = F.log_softmax(batch_preds_t, dim=-1)
            loss = kl_loss(input_, target) + torch.sum(img_preds ** 2, dim=-1).mean() + \
                   torch.sum(batch_preds_t ** 2, dim=-1).mean()

            #loss = nn.CrossEntropyLoss()(batch_preds_t, img_preds.softmax(dim=-1)) + \
                   #torch.sum(img_preds ** 2, dim=-1).mean() + \
                   #torch.sum(batch_preds_t ** 2, dim=-1).mean()
            print('val', y)
            print('val', img_preds.softmax(dim=-1)[:, 1])
            print('val', batch_preds_t.softmax(dim=-1)[:, 1])

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
            img_preds, batch_preds_t, batch_preds_s, \
            batch_nodes_t, batch_nodes_s, batch_anomaps = self.model(x, self.pad_green)

            scores = []
            for anomap in batch_anomaps:
                self.min_max.update(anomap)
                score, _ = anomap.max(dim=0)
                scores.append(score)
            scores = torch.cat(scores)
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
