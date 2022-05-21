from utils.common.common import load_config
from utils.common.project_paths import GetPaths
import torch
import numpy as np
import random


def global_setting(config_file='cfg.yaml'):
    config = load_config(GetPaths().get_configs_folder(config_file))
    torch.manual_seed(config["SEED"])
    np.random.seed(config["SEED"])
    random.seed(config["SEED"])
    torch.cuda.manual_seed(config["SEED"])
    torch.cuda.manual_seed_all(config["SEED"])  # if use multi-GPU

    # 나중에 재현할 때 쓰는 코드
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    if config['TRAIN']['DEVICE'] is not None:
        device = config['TRAIN']['DEVICE']
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return config, device
