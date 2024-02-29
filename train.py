import argparse
import torch.cuda
from utils import *
from models.trainer import *
from train_tools.dataset import *
from train_tools.metrics import *
from train_tools.callbacks import *
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='../datasets/HBM/HBM-AfterManualJudge/type3')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--val_ratio', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=20)
    opt, _ = parser.parse_known_args()
    return opt

def main():
    opt = parse_opt()
    config = get_config()
    build_dirs(config, opt)

    #train_tools
    set_seed(0)
    model = RegionClip(config)
    dataset = HBMDataModule(opt, model._transform)
    if os.path.exists(config['ckpt_dir']):
        model.load_from_checkpoint(config['ckpt_dir'])
        print('load weights successfully')

    #callbacks
    earlystop = EarlyStopping(monitor="loss:", patience=3, mode="min")
    modelckpt = ModelCheckpoint(monitor='loss:',
                dirpath = opt.ckpt_dir,
                filename = config['file_name'],
                mode='min',
            )
    visualizer = Visualizer(dataset, config)
    #train
    trainer = L.Trainer(
        max_epochs= opt.num_epochs,
        callbacks=[modelckpt, visualizer],
        accelerator="gpu",
        logger= CSVLogger(config['output_log_dir']),
        check_val_every_n_epoch=2,
    )
    trainer.fit(model=model, datamodule=dataset)

if __name__ == '__main__':
    main()