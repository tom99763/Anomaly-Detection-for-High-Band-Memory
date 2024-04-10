from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import torch
from skimage.segmentation import mark_boundaries

mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].cuda()
std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].cuda()

class Visualizer(Callback):
    def __init__(self, dataset, config):
        self.state = {"epochs": 0}
        self.config = config
        sample, label = next(iter(dataset.val_dataloader())) #(b, 3, h, w)
        self.sample = sample.cuda()
        self.label = label.cuda()
    def on_validation_start(self, trainer, pl_module):
        trainer.model.model.encoder.eval()
        print(self.label)
        self.state["epochs"] += 1
        with torch.no_grad():
            ft, fs = trainer.model.model(self.sample, train=True)
            x = ft['layer0']
            xs = ft['layer0']
            plt.tight_layout()
            plt.savefig(f"{self.config['output_img_dir']}/{self.state['epochs']}.png")






