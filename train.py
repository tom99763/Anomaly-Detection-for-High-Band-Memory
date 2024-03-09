import argparse
import torch.cuda
from utils import *
from models.trainer import *
from models.trainer_semi import *
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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--learning_type', type=str, default='semi_sup')
    opt, _ = parser.parse_known_args()
    return opt

def main(config, opt):
    build_dirs(config, opt)

    #train_tools
    set_seed(0)
    if os.path.exists(config['ckpt_dir']):
        if opt.learning_type == 'sup':
            model = RegionClip.load_from_checkpoint(config['ckpt_dir'], config=config)
        elif opt.learning_type == 'semi_sup':
            model = RegionClipSemi.load_from_checkpoint(config['ckpt_dir'], config=config)
        else:
            raise Exception('specify training type')
        print('load weights successfully')
    else:
        if opt.learning_type == 'sup':
            model = RegionClip(config)
        elif opt.learning_type== 'semi_sup':
            model = RegionClipSemi(config)
        else:
            raise Exception('specify training type')
        print('no weights')
    dataset = HBMDataModule(opt, model._transform)

    #callbacks
    earlystop = EarlyStopping(monitor="val_aupr", patience=5, mode="max")
    modelckpt = ModelCheckpoint(monitor='val_aupr',
                dirpath = f"{opt.ckpt_dir}/{opt.learning_type}/{config['gnn']['net_type']}",
                filename = config['file_name'],
                mode='max',
            )
    visualizer = Visualizer(dataset, config)
    #train
    trainer = L.Trainer(
        max_epochs= opt.num_epochs,
        callbacks=[earlystop, modelckpt, visualizer],
        accelerator="gpu",
        logger= CSVLogger(config['output_log_dir']),
        check_val_every_n_epoch=2,
    )

    if opt.mode == 'train':
        trainer.fit(model=model, datamodule=dataset)
    else:
        trainer.test(model = model, datamodule=dataset)

if __name__ == '__main__':
    opt = parse_opt()
    config = get_config()
    main(config, opt)

    '''
    gnn_type = ['GAT', 'GCN']
    class_names = ['green dots']
    #class_names = ['dots', 'black dots', 'green dots']
    num_segments = [200, 100, 75]
    #num_segments = [75, 100, 200]
    net_types = ['gnn', 'linear']

    for net_type in net_types:
        config['gnn']['net_type'] = net_type
        for class_name in class_names:
            config['clip']['class_name'] = class_name
            for num_segments_ in num_segments:
                config['superpixel']['numSegments'] = num_segments_
                if net_type != 'gnn':
                    main(config, opt)
                else:
                    for gnn_type_ in gnn_type:
                        config['gnn']['gnn_type'] = gnn_type_
                        main(config, opt)
    '''

