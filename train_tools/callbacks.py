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
            recon_imgs, recon_objects, masks = trainer.model.model(self.sample)
            recon_objects = recon_objects * masks
            # (b, 3, h, w)
            # (b, k, 3, h, w)
            batch_size = recon_objects.shape[0]
            num_slots = recon_objects.shape[1]
            fig, ax = plt.subplots(ncols=4, nrows=2 + recon_objects.shape[1], figsize= (24, 24))
            for i in range(4):
                img = self.sample[i] #* std + mean
                img = img.permute(1, 2, 0).cpu()
                recon_img = recon_imgs[i] * 0.5 + 0.5
                recon_img = recon_img.permute(1, 2, 0).cpu()
                recon_object = recon_objects[i] * 0.5 + 0.5
                recon_object = recon_object.permute(0, 2, 3, 1).cpu()
                ax[0, i].imshow(img)
                ax[0, i].axis('off')
                ax[1, i].imshow(recon_img)
                ax[1, i].axis('off')
                for j in range(num_slots):
                    ax[2 + j, i].imshow(recon_object[j])
                    ax[2 + j, i].axis('off')
            plt.tight_layout()
            plt.savefig(f"{self.config['output_img_dir']}/{self.state['epochs']}.png")






