import argparse
from utils import *
from models.trainer import *
from train.dataset import *
from train.metrics import *
from train.callbacks import *
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='../datasets/HBM/type3')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--val_ratio', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    opt, _ = parser.parse_known_args()
    return opt

def main():
    torch.manual_seed(0)
    opt = parse_opt()
    config = get_config()
    build_dirs(config, opt)

    #train
    model = RegionClip(config)
    dataset = HBMDataModule(opt, model._transform)
    if os.path.exists(config['ckpt_dir']):
        model.load_from_checkpoint(config['ckpt_dir'])
        print('load weights successfully')
    model.train()
    trainer = L.Trainer(
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, mode="min")
        ],
        default_root_dir=opt.ckpt_dir,
        accelerator="gpu"
    )


if __name__ == 'main':
    main()