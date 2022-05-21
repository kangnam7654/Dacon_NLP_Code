import yaml


def load_config(cfg_path):
    return yaml.full_load(open(cfg_path, 'r', encoding='utf-8-sig'))

