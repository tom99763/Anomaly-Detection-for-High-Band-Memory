import torch
import torch.nn as nn
import lightning as L

class RegionClip(L.LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self):
        pass

    def validation_step(self):
        pass

    def configure_optimizers(self):
        pass
    @property
    def trainer_arguments(self):
        """Set model-specific trainer arguments."""
        return {}
