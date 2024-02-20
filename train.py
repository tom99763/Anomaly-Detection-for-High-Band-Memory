import argparse
from utils import *
from models.trainer import *
from train.dataset import *
from train.metrics import *
from train.callbacks import *

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    opt, _ = parser.parse_known_args()
    return opt

def main():
    torch.manual_seed(0)
    opt = parse_opt()
    config = get_config()
    build_dirs(config, opt)
    trainer = RegionClip(config)



if __name__ == 'main':
    main()