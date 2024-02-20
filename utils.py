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
    gnn_type = config['gnn']['gnn_type']
    file_name = f'{model_name}_{class_name}_{num_prompts}_{num_segments}_{gnn_type}'
    config['file_name'] = file_name
    output_dir = f'{opt.output_dir}/{file_name}'
    ckpt_dir = f'{opt.ckpt_dir}/{file_name}.ckpt'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config['output_dir'] = output_dir
    config['ckpt_dir'] = ckpt_dir
    output_img_dir = f'{output_dir}/img'
    output_log_dir = f'{output_dir}/log'
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_log_dir):
        os.makedirs(output_log_dir)
    config['output_img_dir'] = output_img_dir
    config['output_log_dir'] = output_log_dir