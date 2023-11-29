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
import glob
from PIL import Image


class dataset(torch.utils.data.Dataset):
    def __init__(self,file_list,transform=None):
        self.file_list = file_list
        self.transform = transform
        
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    
    def __getitem__(self,idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        label = img_path.split("\\")[1].split("_")[0]
        if label == 'leopard':
            label=1
        elif label == 'tiger':
            label=0
            
        return img_transformed,label

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3, padding=0,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        
        self.fc1 = nn.Linear(3*3*64,10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10,2)
        self.relu = nn.ReLU()
        
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

def data_loader(batch_size: int) -> torch.utils.data.DataLoader:

    "creating data loaders for model training"

    train_list, test_list, validate_list = return_lists()
    train_transforms, test_transforms, validate_transforms = data_augmentation()

    train_data = dataset(train_list, transform=train_transforms)
    test_data = dataset(test_list, transform=test_transforms)
    validate_data = dataset(validate_list, transform=validate_transforms)

    train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
    test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(dataset = validate_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, validate_loader
    

def data_augmentation() -> transforms.transforms.Compose:
    "Changes image parameters"
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
    device = "cpu"
    lr = 0.001 # learning_rate
    batch_size = 100 # we will use mini-batch method
    epochs = 10 # How much to train a model

    train_loader, test_loader, validate_loader = data_loader(batch_size)
    model = Cnn().to(device)
    
    optimizer = optim.Adam(params = model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            
            output = model(data)
            loss = criterion(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = ((output.argmax(dim=1) == label).float().mean())
            epoch_accuracy += acc/len(train_loader)
            epoch_loss += loss/len(train_loader)
            
        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
        
        
        with torch.no_grad():
            epoch_val_accuracy=0
            epoch_val_loss =0
            for data, label in validate_loader:
                data = data.to(device)
                label = label.to(device)
                
                val_output = model(data)
                val_loss = criterion(val_output,label)
                
                
                acc = ((val_output.argmax(dim=1) == label).float().mean())
                epoch_val_accuracy += acc/ len(validate_loader)
                epoch_val_loss += val_loss/ len(validate_loader)
                
            print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))



if __name__ == "__main__":
    main()