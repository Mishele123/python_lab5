import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile


def get_list_dirs() -> None:
    train_dir = "train_dir"
    print(os.listdir(train_dir))



def main() -> None:
    get_list_dirs()

if __name__ == "__main__":
    main()