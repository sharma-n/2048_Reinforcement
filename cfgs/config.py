from easydict import EasyDict
from pathlib import Path
import yaml

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
        config.update(EasyDict(new_config))

    return config


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0