import argparse
import torch.cuda
from utils import *
from models.trainer import *
from models.trainer_semi import *
from models.trainer_lp import *
from models.trainer_stfpm import *
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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--learning_type', type=str, default='st')
    opt, _ = parser.parse_known_args()
    return opt

def main(config, opt):
    build_dirs(config, opt)
    mode = config['st']['mode']
    #train_tools
    set_seed(0)

    if opt.learning_type == 'stfpm':
        if os.path.exists(config['ckpt_dir']):
            model = STFPM.load_from_checkpoint(config['ckpt_dir'], config=config)
            print('load weights')
        else:
            model = STFPM(config)
            print('no weights')

    elif opt.learning_type == 'st':
        if os.path.exists(config['ckpt_dir_t']):
            model = RegionClipLP.load_from_checkpoint(config['ckpt_dir_t'], config=config)
            print('load teacher weights')
        else:
            model = RegionClipLP(config)

        if os.path.exists(config['ckpt_dir_s']):
            return
            print('load student weights')
            model = RegionClipLP.load_from_checkpoint(config['ckpt_dir_s'], config=config)


    dataset = HBMDataModule(opt)

    if mode == 'student':
        earlystop = EarlyStopping(monitor="val_f1_score", patience=5, mode="max")
        modelckpt = ModelCheckpoint(monitor='val_f1_score',
                                    dirpath=f"{opt.ckpt_dir}/{opt.learning_type}/"
                                            f"{opt.dataset_dir.split('/')[-1]}/"
                                            f"{config['gnn']['net_type']}",
                                    filename=f"{config['file_name']}_s",
                                    mode='max')
        trainer = L.Trainer(
            max_epochs=opt.num_epochs,
            callbacks=[earlystop, modelckpt],
            accelerator="gpu",
            logger=CSVLogger(config['output_log_dir']),
            check_val_every_n_epoch=4,
        )

    elif mode == 'teacher':
        earlystop = EarlyStopping(monitor="val_loss", patience=2, mode="min")
        modelckpt = ModelCheckpoint(monitor='val_loss',
                                    dirpath=f"{opt.ckpt_dir}/{opt.learning_type}/"
                                            f"{opt.dataset_dir.split('/')[-1]}/"
                                            f"{config['gnn']['net_type']}",
                                    filename=f"{config['file_name']}_t",
                                    mode='min')
        trainer = L.Trainer(
            max_epochs=opt.num_epochs,
            callbacks=[earlystop, modelckpt],
            accelerator="gpu",
            logger=CSVLogger(config['output_log_dir']),
            check_val_every_n_epoch=2,
        )



    if opt.mode == 'train':
        trainer.fit(model=model, datamodule=dataset)
    else:
        trainer.test(model = model, datamodule=dataset)

if __name__ == '__main__':
    opt = parse_opt()
    config = get_config()

    num_segments = [300, 100, 200, 75]
    gnn_type = ['GAT', 'GCN']
    net_type = ['gnn', 'linear']

    for net_type_ in net_type:
        config['gnn']['net_type'] = net_type_
        for num_segment_ in num_segments:
            config['superpixel']['numSegments'] = num_segment_
            if net_type_ =='gnn':
                for gnn_type_ in gnn_type:
                    config['gnn']['gnn_type'] = gnn_type_
                    config['st']['mode'] = 'teacher'
                    main(config, opt)
                    config['st']['mode'] = 'student'
                    main(config, opt)
            else:
                config['st']['mode'] = 'teacher'
                main(config, opt)
                config['st']['mode'] = 'student'
                main(config, opt)





