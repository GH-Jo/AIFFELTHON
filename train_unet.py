#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import sampler
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import numpy as np
from torch.utils.data import DataLoader
from copy import copy


def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
set_seed(42)


import wandb
wandb.login()

##############################
#####  Prepare Datasets  #####
##############################
df = pd.read_csv('data/train_info.csv')
test_df = pd.read_csv('data/test_info.csv')

df['imgpath'] = df['prefix'] + "images/" + df['filename']
df['mskpath'] = df['prefix'] + "masks/" + df['filename']
test_df['imgpath'] = test_df['prefix'] + "images/" + test_df['filename']
test_df['mskpath'] = test_df['prefix'] + "masks/" + test_df['filename']

include_normal = True
if include_normal:
    train_scratch_df = df[('scratch'==df['prefix'].str[:7]) & ('train'== df['prefix'].str[-6:-1]) ]
    train_dent_df = df[('dent'==df['prefix'].str[:4]) & ('train'==df['prefix'].str[-6:-1])]
    train_spacing_df = df[('spacing'==df['prefix'].str[:7]) & ('train'==df['prefix'].str[-6:-1])]
else:
    train_scratch_df = df[df['scratch'] & ('train'== df['prefix'].str[-6:-1])]
    train_dent_df = df[df['dent'] & ('train'== df['prefix'].str[-6:-1])]
    train_spacing_df = df[df['spacing'] & ('train'==df['prefix'].str[-6:-1])]
    
test_scratch_df = test_df['scratch'==test_df['prefix'].str[:7]]
test_dent_df = test_df['dent'==test_df['prefix'].str[:4]]
test_spacing_df = test_df['spacing'==test_df['prefix'].str[:7]]

valid_scratch_df = df[('scratch'==df['prefix'].str[:7]) & ('valid'== df['prefix'].str[-6:-1]) ]
valid_dent_df = df[('dent'==df['prefix'].str[:4]) & ('valid'==df['prefix'].str[-6:-1])]
valid_spacing_df = df[('spacing'==df['prefix'].str[:7]) & ('valid'==df['prefix'].str[-6:-1])]


class SocarDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None): 
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        return: image, mask
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # 1. create path
        img_path = self.root_dir + self.df.iloc[idx]['imgpath']
        msk_path = self.root_dir + self.df.iloc[idx]['mskpath']
        
        # 2. load image
        image = Image.open(img_path)
        mask = Image.open(msk_path)
        if self.transform:
            image = self.transform[0](image)
            mask = self.transform[1](mask)
        return image, mask

from torchvision import transforms
data_transformer = transforms.Compose(
    [#transforms.ColorJitter(contrast=0.3, brightness=0.3, saturation=0.3, hue=0.3),
     transforms.Resize((512, 512)),
     transforms.ToTensor()]
)
mask_transformer = transforms.Compose(
    [transforms.Resize((512, 512)),
     transforms.ToTensor()]
)

root_dir = '/home/lkj004124/accida_segmentation_dataset_v1/'
dataset_train_scratch = SocarDataset(train_scratch_df, root_dir, (data_transformer, mask_transformer))
dataset_train_dent = SocarDataset(train_dent_df, root_dir, (data_transformer, mask_transformer))
dataset_train_spacing = SocarDataset(train_spacing_df, root_dir, (data_transformer, mask_transformer))

dataset_test_scratch = SocarDataset(test_scratch_df, root_dir, (mask_transformer, mask_transformer))
dataset_test_dent = SocarDataset(test_dent_df, root_dir, (mask_transformer, mask_transformer))
dataset_test_spacing = SocarDataset(test_spacing_df, root_dir, (mask_transformer, mask_transformer))

dataset_valid_scratch = SocarDataset(valid_scratch_df, root_dir, (mask_transformer, mask_transformer))
dataset_valid_dent = SocarDataset(valid_dent_df, root_dir, (mask_transformer, mask_transformer))
dataset_valid_spacing = SocarDataset(valid_spacing_df, root_dir, (mask_transformer, mask_transformer))

bs = 1
loader_train_scratch = DataLoader(dataset_train_scratch, batch_size = bs, shuffle=True, num_workers=1)
loader_train_dent = DataLoader(dataset_train_dent, batch_size = bs, shuffle=True, num_workers=1)
loader_train_spacing = DataLoader(dataset_train_spacing, batch_size = bs, shuffle=True, num_workers=1)

loader_test_scratch = DataLoader(dataset_test_scratch, batch_size = 1, shuffle=False, num_workers=1)
loader_test_dent = DataLoader(dataset_test_dent, batch_size = 1, shuffle=False, num_workers=1)
loader_test_spacing = DataLoader(dataset_test_spacing, batch_size = 1, shuffle=False, num_workers=1)

loader_valid_scratch = DataLoader(dataset_valid_scratch, batch_size = 1, shuffle=True, num_workers=1)
loader_valid_dent = DataLoader(dataset_valid_dent, batch_size = 1, shuffle=True, num_workers=1)
loader_valid_spacing = DataLoader(dataset_valid_spacing, batch_size = 1, shuffle=True, num_workers=1)




################
###   Utils  ###
################
EPS = 1e-8
def calc_IoU(pred, true): # for 2-or-more-channel
    u = torch.argmax(pred, axis=1)
    i = torch.argmax(true, axis=1)
    return (u*i).sum() / ((u|i).sum()+EPS)


#################
###   Model   ###
#################
import segmentation_models_pytorch as smp
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",    
    in_channels=3,                  
    classes=2,                      
).to(device)

#  You can choose task by index
tasks = ["scratch", "dent", "spacing"]

loader_train = globals(f'loader_train_{tasks[0]}') 
loader_valid = globals(f'loader_valid_{tasks[0]}') 
loader_test  = globals(f'loader_test_{tasks[0]}') 


################
###  Train  ####
################
LR = 0.0001
EPOCHS = 25
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = LR)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)

wandb.init(
    project="Basic-scratch",
    config={
        "learning_rate":LR,
        "scheduler":"LambdaLR",
        "architecture":"U-Net",
        "epochs": EPOCHS,
        "backbone": "efficientnet-b7"
    }
)



for epoch in range(EPOCHS):  # loop over the dataset multiple times
    running_loss = 0.0
    print('train')
    IOU = 0
    model.train()
    for i, data in tqdm(enumerate(loader_train, 0)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels[:,:1].to(device)
        labels_0 = 1 - labels
        labels = torch.cat([labels_0, labels], axis=1)
        labels = torch.argmax(labels, axis=1)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        IOU += calc_IoU(outputs, labels) 
        
        if i % 200 == 199:    
            metrics = {"train/train_loss": running_loss / 200,
                       "train/epoch": epoch+1,
                       "train/IOU": IOU / 200,
                       "train/lr": optimizer.param_groups[0]['lr']}
            wandb.log(metrics)
            running_loss = 0.0
            IOU = 0 

    scheduler.step()
    print('validation')
    model.eval()
    with torch.no_grad():
        val_loss = 0
        PA = 0 
        IOU = 0
        for i, data in tqdm(enumerate(loader_valid, 0)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels[:,:1].to(device)
            labels_0 = 1 - labels
            labels = torch.cat([labels_0, labels], axis=1)
            labels = torch.argmax(labels, axis=1)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            IOU += calc_IoU(outputs, labels)
        IOU /= len(loader_valid.dataset)
        val_loss = val_loss / len(loader_valid.dataset)
        print(f'val/loss: {val_loss:4f}')
        wandb.log({'val/loss': val_loss,
                 'val/IOU': IOU})
print('Finish Training')
wandb.finish()


################
####  Test  ####
################
model.eval()
with torch.no_grad():
    total_loss = 0
    running_loss = 0.0
    n_class = 2
    IOU = 0
    for i, data in tqdm(enumerate(loader_test, 0)):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels[:,:1].to(device)
        labels_0 = 1 - labels
        labels = torch.cat([labels_0, labels], axis=1)
        labels = torch.argmax(labels, axis=1)
        
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        total_loss += loss
        running_loss += loss.item()
        
        IOU += calc_IoU(outputs, labels)
    IOU /= len(loader_test.dataset)
    print(f'IoU score: {IOU}')
    
    print('Finish Testing')



