from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import torch
from skimage.segmentation import mark_boundaries

class Visualizer(Callback):
    def __init__(self, dataset, config):
        self.state = {"epochs": 0}
        self.config = config
        self.sample = next(iter(dataset.val_dataloader())).to('cuda') #(b, 3, h, w)
    def on_train_epoch_end(self, trainer, pl_module):
        self.state["epochs"] += 1
        with torch.no_grad():
            batch_preds, batch_regions = trainer.model.model(self.sample) #[(N, 2), (h, w)]
        batch_size, _ , h, w = self.sample.shape
        fig, ax = plt.subplots(ncols = batch_size, nrows = 3, figsize = (12, 12))
        for i in range(batch_size):
            image = self.sample[i].permute(1, 2, 0).cpu()
            mask = torch.zeros((h, w))
            preds = batch_preds[i].cpu()
            regions = batch_regions[i].cpu() #(h, w)
            bound = mark_boundaries(image.numpy(), regions.numpy() + 1)
            for j in range(regions.unique().shape[0]):
                region_pred = preds[j].argmax() #(2, )
                idx = regions==j
                mask[idx] = region_pred.float()
            ax[0, i].imshow(image)
            ax[0, i].axis('off')
            ax[1, i].imshow(bound)
            ax[1, i].axis('off')
            ax[2, i].imshow(image)
            ax[2, i].imshow(mask, alpha=0.7)
            ax[2, i].axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.config['output_img_dir']}/{self.state['epochs']}.png")
