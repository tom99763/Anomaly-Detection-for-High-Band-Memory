import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
from .model_stfpm import *
from train_tools.metrics import *
from .losses import *
from torchmetrics.classification import BinaryF1Score
import torch.nn.functional as F
from .threshold import *

class STFPM(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = STFPM_Model(config)
        self.config = config

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
        self.model.student.train()
        self.model.teacher.eval()
        x, y = batch
        y = y.cuda()
        ft, fs = self.model(x, True)
        loss = stfpm_loss(ft, fs)
        self.step += 1
        self.log_dict({'loss:': loss.item(),
                       },
                      on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch):
        self.model.teacher.eval()
        self.model.student.eval()
        x, y = batch
        y = y.cuda()

        anomap = self.model(x)
        score, _ = anomap.flatten(1, 3).max(dim=-1)

        #update score
        self.min_max.update(anomap)
        self.f1_adapt.update(score, y)
        score = normalize(score, self.min, self.max, self.thr)

        # metrics
        self.auroc.update(score, y)
        self.aupr.update(score, y)
        self.f1_score.update(score, y)

    def on_validation_epoch_end(self):
        auroc = self.auroc.compute()
        aupr = self.aupr.compute()
        f1_score = self.f1_score.compute()
        sum_score = auroc + aupr + f1_score
        self.log_dict(
            {'val_auroc': auroc, 'val_aupr': aupr, 'val_f1_score': f1_score,
             'val_min': self.min, 'val_max': self.max, 'f1_thr': self.thr, 'val_sum_score': sum_score
             })
        self.auroc.reset()
        self.aupr.reset()
        self.f1_score.reset()
        self.thr = self.f1_adapt.compute()
        self.min, self.max = self.min_max.compute()
        self.f1_adapt.reset()
        self.min_max.reset()


    def test_step(self, batch):
        self.model.teacher.eval()
        self.model.student.eval()
        x, y = batch
        y = y.cuda()

        anomap = self.model(x)
        score, _ = anomap.flatten(1, 3).max(dim=-1)
        score = normalize(score, self.min, self.max, self.thr)

        # metrics
        self.auroc.update(score, y)
        self.aupr.update(score, y)
        self.f1_score.update(score, y)

    def on_test_end(self):
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
        optimizer= optim.SGD(
            params=self.model.student.parameters(),
            lr=0.4,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.001,
        )
        return {"optimizer": optimizer}

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

