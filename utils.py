import os
import yaml

def get_config():
    config_path = './configs/RegionClip.yaml'
    with open(config_path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)
def build_dirs(config, opt):
    model_name = config['clip']['model_name']
    class_name = config['clip']['class_name']
    num_prompts = config['clip']['num_prompts']
    num_segments = config['superpixel']['numSegments']
    file_name = f'{model_name}_{class_name}_{num_prompts}_{num_segments}'
    output_dir = f'{opt.output_dir}/{file_name}'
    ckpt_dir = f'{opt.ckpt_dir}/{file_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    config['output_dir'] = output_dir
    config['ckpt_dir'] = ckpt_dir