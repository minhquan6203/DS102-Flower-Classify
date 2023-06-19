import argparse
import torch
from utils.get_config import get_config
from task.classify_task import Classify_Task
from task.clustering_task import Clustering_Task
parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, required = True)

args = parser.parse_args()

config = get_config(args.config_file)

torch.manual_seed(1234)

if config.task == 'classify':
    task = Classify_Task(config)
    # task.training()
    # task.evaluate() 
    task.demo()
if config.task == 'clustering':
    task = Clustering_Task(config)
    task.training_and_eval()
    