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


def data_augmentation() -> transforms.transforms.Compose:
    train_transforms =  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    validate_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])


    test_transforms = transforms.Compose([   
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])
    
    
    return train_transforms, validate_transforms, test_transforms


def return_lists() -> list:
    
    "the function returns an array containing path to the images (split datas)"
    from sklearn.model_selection import train_test_split

    train_dir = "train_dir"

    train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
    train_list, temp_list = train_test_split(train_list, test_size=0.2)
    test_list, validate_list = train_test_split(temp_list, test_size=0.5)

    # print(train_list[0].split("\\")[1].split("_")[0])

    return train_list, test_list, validate_list


def main() -> None:
    data_augmentation()

if __name__ == "__main__":
    main()