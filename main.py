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
import glob
from PIL import Image
import random




def return_lists() -> list:
    "the function returns an array containing the names of the images"
    train_dir = "train_dir"
    test_dir = "test_dir"
    validate_dir = "validate_dir"

    train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
    test_list = glob.glob(os.path.join(test_dir,'*.jpg'))
    validate_list = glob.glob(os.path.join(validate_dir,'*.jpg'))
    

    print(train_list[0].split("\\")[1].split("_")[0])

    return train_list, test_list, validate_list


def main() -> None:
    return_lists()

if __name__ == "__main__":
    main()