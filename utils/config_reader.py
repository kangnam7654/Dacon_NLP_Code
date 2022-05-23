from utils.common.project_paths import GetPaths
import yaml


def cfg_load(file_name='cfg.yaml'):
    cfg_file = GetPaths.get_configs_folder(file_name)
    with open(cfg_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':
    cfg = cfg_load()
