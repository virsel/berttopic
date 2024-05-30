
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from data import get_train_test_split
from src.moduls.config import get_default_config

from src.moduls.custom_logging import set_logging, Logger
from src.moduls.model import get_model

set_logging()
import torch
import numpy as np
import random
from src.moduls.trainer import get_trainer
import os


# Set a specific seed value for reproducibility
seed_value = 42  # Choose any integer you want

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"



