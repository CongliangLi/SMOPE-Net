from pprint import pprint
from .configure import Config
import os
import time

configure_name = 'config.json'

config = Config.get_configure(configure_name)


def abs_join_path(path1, path2, is_create=True):
    if path1 is None or path2 is None:
        return None
    join_path = os.path.abspath(os.path.join(path1, path2))
    if is_create and not os.path.exists(join_path):
        os.makedirs(join_path)
    return join_path


# convert relative path to absolute path
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
cfg = config[config["phase"]]
cfg["output_dir"] = config[config["phase"]]["output_dir"]
if cfg["resume"] != "":
    cfg["resume"] = abs_join_path(ROOT_DIR, cfg["resume"])
else:
    cfg["resume"] = None
    
cfg["model_folder"] = os.path.join(config["dataset_path"], "model_{}".format(config["dataset_path"].split("/")[-1]))

cfg["model_path"] = [os.path.join(cfg["model_folder"], i)for i in config["model"]["model_name"]]

print('loading configure: ' + configure_name)
print("========")
pprint(config)
print('config done')
print("========")
