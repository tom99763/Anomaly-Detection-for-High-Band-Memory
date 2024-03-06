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
    parser.add_argument('--dataset_dir', type=str, default='../datasets/HBM/HBM-AfterManualJudge/type2')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--val_ratio', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--mode', type=str, default='test')
    opt, _ = parser.parse_known_args()
    return opt

def main(config, opt):
    build_dirs(config, opt)

    #train_tools
    set_seed(0)
    if os.path.exists(config['ckpt_dir']):
        model = RegionClip.load_from_checkpoint(config['ckpt_dir'], config=config)
        print('load weights successfully')
    else:
        model = RegionClip(config)
        print('no weights')
    dataset = HBMDataModule(opt, model._transform)

    #callbacks
    earlystop = EarlyStopping(monitor="val_auroc", patience=5, mode="max")
    modelckpt = ModelCheckpoint(monitor='val_auroc',
                dirpath = f"{opt.ckpt_dir}/{config['gnn']['net_type']}",
                filename = config['file_name'],
                mode='max',
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

    if opt.mode == 'train':
        trainer.fit(model=model, datamodule=dataset)
    else:
        trainer.test(model = model, datamodule=dataset)

if __name__ == '__main__':
    opt = parse_opt()
    config = get_config()

    gnn_type = ['GAT', 'GCN']
    share_prompt = [True, False]
    linear_probe = [False, True]
    num_segments = [75, 100, 200]
    num_prompts = [0, 4, 8, 12]

    config['gnn']['net_type'] = 'gnn'
    for num_segments_ in num_segments:
        for gnn_type_ in gnn_type:
            for linear_probe_ in linear_probe:
                if linear_probe_:
                    config['superpixel']['numSegments'] = num_segments_
                    config['gnn']['gnn_type'] = gnn_type_
                    config['prompt']['linear_probe'] = linear_probe_
                    config['clip']['num_prompts'] = 0
                    config['prompt']['share_prompt'] = True
                    main(config, opt)
                else:
                    for num_prompts_ in num_prompts:
                        for share_prompt_ in share_prompt:
                            config['superpixel']['numSegments'] = num_segments_
                            config['gnn']['gnn_type'] = gnn_type_
                            config['prompt']['linear_probe'] = linear_probe_
                            config['clip']['num_prompts'] = num_prompts_
                            config['prompt']['share_prompt'] = share_prompt_
                            main(config, opt)

    config['gnn']['net_type'] = 'linear'
    for num_segments_ in num_segments:
        for linear_probe_ in linear_probe:
            if linear_probe_:
                config['superpixel']['numSegments'] = num_segments_
                config['prompt']['linear_probe'] = linear_probe_
                config['clip']['num_prompts'] = 0
                config['prompt']['share_prompt'] = True
                main(config, opt)
            else:
                for num_prompts_ in num_prompts:
                    for share_prompt_ in share_prompt:
                        config['superpixel']['numSegments'] = num_segments_
                        config['prompt']['linear_probe'] = linear_probe_
                        config['clip']['num_prompts'] = num_prompts_
                        config['prompt']['share_prompt'] = share_prompt_
                        main(config, opt)         

    config['gnn']['net_type'] = 'none'
    for num_segments_ in num_segments:
        for linear_probe_ in linear_probe:
            if linear_probe_:
                config['superpixel']['numSegments'] = num_segments_
                config['prompt']['linear_probe'] = linear_probe_
                config['clip']['num_prompts'] = 0
                config['prompt']['share_prompt'] = True
                main(config, opt)
            else:
                for num_prompts_ in num_prompts:
                    if num_prompts_ == 0:
                        continue
                    for share_prompt_ in share_prompt:
                        config['superpixel']['numSegments'] = num_segments_
                        config['prompt']['linear_probe'] = linear_probe_
                        config['clip']['num_prompts'] = num_prompts_
                        config['prompt']['share_prompt'] = share_prompt_
                        main(config, opt)

