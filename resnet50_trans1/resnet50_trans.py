#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division

# In[1]:



# 
# 전이학습(Transfer Learning)
# ========================
# -  **합성곱 신경망의 미세조정(finetuning)**: 무작위 초기화 대신, 신경망을
#    ImageNet 1000 데이터셋 등으로 미리 학습한 신경망으로 초기화합니다. 학습의 나머지
#    과정들은 평상시와 같습니다.
# -  **고정된 특징 추출기로써의 합성곱 신경망**: 여기서는 마지막에 완전히 연결
#    된 계층을 제외한 모든 신경망의 가중치를 고정합니다. 이 마지막의 완전히 연결된
#    계층은 새로운 무작위의 가중치를 갖는 계층으로 대체되어 이 계층만 학습합니다.



# License: BSD
# Author: Sasank Chilamkurthy


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from PIL import Image
from tqdm import tqdm
import random

cudnn.benchmark = True
#plt.ion()   # 대화형 모드


# 랜덤시드 설정하기

# In[3]:


def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

'''
set_seed(1004)


# In[4]:


EPOCHS = 30
_batch_size = 32
_dataset_folder_path = '../dataset1'
_train_info_csv_path = os.path.join(_dataset_folder_path,'socar_train_info_cnn.csv')
_test_info_csv_path = os.path.join(_dataset_folder_path,'socar_test_info_cnn.csv')
'''

# # DataSet 생성

# In[5]:


class MyDataset(torch.utils.data.Dataset):
    """socar (spacing,dent,scratch) image dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): csv 파일의 경로
            root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):

        prefix = self.data_frame.iloc[idx].prefix
        fname = self.data_frame.iloc[idx].filename
        img_name = os.path.join(self.root_dir, os.path.join(os.path.join(prefix,'images'),fname))
        #msk_name = os.path.join(self.root_dir, os.path.join(os.path.join(prefix,'masks'),fname))
        img = Image.open(img_name)
        #msk= Image.open(msk_name)
        label_list = self.data_frame.iloc[idx].label
        label=0
        if label_list==True:
            label=1        

        if self.transform:
            img= self.transform(img)
            #msk= self.transform(msk)

        return img,label,img_name

        

class MyDataset2(torch.utils.data.Dataset):
    """socar (spacing,dent,scratch) image dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): csv 파일의 경로
            root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):

        
        fname = self.data_frame.iloc[idx].filename
        img_name = os.path.join(self.root_dir, os.path.join(os.path.join('train','images'),fname))
        #msk_name = os.path.join(self.root_dir, os.path.join(os.path.join(prefix,'masks'),fname))
        img = Image.open(img_name)
        #msk= Image.open(msk_name)
        label_list = self.data_frame.iloc[idx, 1:4].to_list()
        label=0
        if label_list[0]==True:
            label=1
        if label_list[1]==True:
            label=1
        if label_list[2]==True:
            label=1
        

        if self.transform:
            img= self.transform(img)
            #msk= self.transform(msk)

        return img,label

        


# 데이터 불러오기
# ---------------
# 
# 데이터를 불러오기 위해 torchvision과 torch.utils.data 패키지를 사용하겠습니다.
# 
# 여기서 풀고자 하는 문제는 **정상** 과 **파손** 을 분류하는 모델을 학습하는 것입니다.
# 정상 이미지 와 파손 이미지 각각의 학습용 이미지는 대략 3000장 정도 있고, 개의 검증용 이미지가
# 있습니다. 일반적으로 맨 처음부터 학습을 한다면 이상 일반화하기에는 아주 작은
# 데이터셋입니다. 하지만 우리는 전이학습을 할 것이므로, 일반화를 제법 잘 할 수 있을
# 것입니다.
# 

# In[6]:



# 학습을 위해 일반화(normalization)
# 검증을 위한 일반화
'''
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((512,512),interpolation=torchvision.transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=(0,180)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((512,512),interpolation=torchvision.transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


image_datasets = {
    'train': MyDataset(_train_info_csv_path,_dataset_folder_path,data_transforms['train']),
    'val': MyDataset(_test_info_csv_path,_dataset_folder_path,data_transforms['val'])
}
                              

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=_batch_size,
#                                              shuffle=True, num_workers=4)
                                             shuffle=True, num_workers=0)
                                             for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = ['normal','problem']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''

# In[7]:


def save(netG, optimG, epoch, dir_check):
    global device
    if not os.path.exists(dir_check):
        os.makedirs(dir_check)

    torch.save({'netG': netG.state_dict(),
                'optimG': optimG.state_dict()},
                '%s/model_epoch%04d.pth' % (dir_check, epoch))


def load(netG, optimG=None, epoch=None,dir_check='./check'):
    global device
    if not os.path.exists(dir_check):
        epoch = 0
        if optimG is None:
            return netG, epoch
        else:
            return netG, optimG, epoch

    if not epoch:
        ckpt = os.listdir(dir_check)
        ckpt = [f for f in ckpt if f.startswith('model')]
        ckpt.sort()

        epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

    dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_check, epoch), map_location=device)

    print('Loaded %dth network' % epoch)

    if optimG is None:
        netG.load_state_dict(dict_net['netG'])
        return netG, epoch
    else:
        netG.load_state_dict(dict_net['netG'])
        optimG.load_state_dict(dict_net['optimG'])
        return netG, optimG, epoch


# 일부 이미지 시각화하기
# 
# 데이터 증가를 이해하기 위해 일부 학습용 이미지를 시각화해보겠습니다.
# 
# 

# In[8]:


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.


# 학습 데이터의 배치를 얻습니다.
#inputs, classes, _ = next(iter(dataloaders['train']))

# 배치로부터 격자 형태의 이미지를 만듭니다.
#out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])


# 모델 학습하기
# --------------
# 
# 이제 모델을 학습하기 위한 일반 함수를 작성해보겠습니다. 여기서는 다음 내용들을
# 설명합니다:
# 
# -  학습률(learning rate) 관리(scheduling)
# -  최적의 모델 구하기
# 
# 아래에서 ``scheduler`` 매개변수는 ``torch.optim.lr_scheduler`` 의 LR 스케쥴러
# 객체(Object)입니다.
# 
# 

# In[9]:


def train_model(model, criterion, optimizer, scheduler = None, num_epochs=25,dir_check='./check'):
    since = time.time()

    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            pbar = tqdm(dataloaders[phase])
            for inputs, labels, _ in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                pbar.set_description(f"{phase} loss={loss.item():.4f} acc={(torch.sum(preds == labels.data)/_batch_size):.4f}")
                
                
            pbar.close()
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()

            print()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} lr: {optimizer.param_groups[0]['lr']:.4f}")
            metrics={
                phase+' loss':epoch_loss,
                phase+' acc':epoch_acc,
                phase+' lr': optimizer.param_groups[0]['lr']
            }
            if phase == 'train':
                wandb.log(metrics, step = epoch, commit=False)
            else:
                wandb.log(metrics, step = epoch, commit=True)

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                #best_model_wts = copy.deepcopy(model.state_dict())
                save(model, optimizer, epoch, dir_check)             

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 가장 나은 모델 가중치를 불러옴
    #model.load_state_dict(best_model_wts)
    return model


# 모델 예측값 시각화하기
# 
# 일부 이미지에 대한 예측값을 보여주는 일반화된 함수입니다.
# 
# 
# 

# In[10]:


def visualize_model(model, num_images=4):
    was_training = model.training
    model.eval()
    images_so_far = 0
    image_list = []
    

    with torch.no_grad():
        for i, (inputs, labels, ing_name) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                fig = plt.figure(figsize=(15,15))
                ax = plt.subplot(2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]} label: {class_names[labels.cpu().data[j]]}')
                imshow(inputs.cpu().data[j])
                
                plt.fig
                # 오탑이미지. label값이 차이날때 if로 이미지를 따로 폴더에 저장...........
                # 파일 이름만 list로 저장해도  차우 선별가능........
                # 위 dataloader에서 paramater filename을 받아서 저장.
                
                if preds[j] != labels.cpu().data[j]:
                    image_list.append({'name':ing_name[j], 'predicted':preds[j], 'label':labels.cpu().data[j], 'output':outputs[j]})

#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return image_list
                
        model.train(mode=was_training)
        return image_list


# In[11]:


def imshow_grid(inputs, titles, save_file=None):
    size = inputs.size(0)
    col = 4
    row = size // col
    fig = plt.figure(figsize=(int(6*col),int(6*row)))

    for i in range(size):
        inp = inputs[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)

        ax = fig.add_subplot(row,col,i+1)
        ax.set_axis_off()
        ax.set_title(titles[i])
        ax.imshow(inp)
    if save_file is not None:
        plt.savefig(save_file)
        plt.close()
    else:
        plt.show()


# In[12]:


def imshow_one(inputs, titles, save_file=None):
#     size = inputs.size(0)
#     col = 4
#     row = size // col
    fig = plt.figure(figsize=(int(6),int(6)))
    inp = inputs.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

#         ax = fig.add_subplot(row,col,i+1)
#         ax.set_axis_off()
#         ax.set_title(titles[i])
    plt.imshow(inp)
    if save_file is not None:
        plt.savefig(save_file)
        plt.close()
    else:
        plt.show()


# In[13]:


def visualize_predict_save(device, dataloaders, model, class_names, batch_num=1, exp_dir_path='.',sel_dataset='val'):
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for i, (inputs, labels, imgname) in enumerate(dataloaders[sel_dataset]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            bath_size = inputs.size()[0]
            
            for j in range(bath_size):
                if preds[j] != labels.cpu().data[j]:
                    label_titles = [class_names[x] for x in labels]
                    pred_titles = [class_names[x] for x in preds]
                    outputtitle = [x for x in outputs]

                    disp_title = [ f"y:{y} y_exp:{y_exp} val:{y_val[0]:.4f}, {y_val[1]:.4f}" for y,y_exp, y_val in zip(label_titles,pred_titles, outputtitle)]
                    disp_title_log = [ f"{fname} y:{y} y_exp:{y_exp}" for fname,y,y_exp in zip(imgname,label_titles,pred_titles)]
                    imshow_one(inputs=inputs.cpu(),
                               titles=disp_title,
                               save_file= exp_dir_path+'/'+sel_dataset+'_'+str(i*bath_size+j)+'.png')
        model.train(mode=was_training)


# 합성곱 신경망 미세조정(finetuning)
# ----------------------------------
# 
# 미리 학습한 모델을 불러온 후 마지막의 완전히 연결된 계층을 초기화합니다.

# In[14]:

'''
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
# 여기서 각 출력 샘플의 크기는 2로 설정합니다.
# 또는, nn.Linear(num_ftrs, len (class_names))로 일반화할 수 있습니다.
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

# 모든 매개변수들이 최적화되었는지 관찰
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

# 7 에폭마다 0.1씩 학습률 감소
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.4)
# exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer=optimizer_ft,
#                                         lr_lambda=lambda epoch: 0.95 ** epoch,
#                                         last_epoch=-1,
#                                         verbose=False)
'''

# In[ ]:





# 학습 및 평가하기
# ----------------------

# In[15]:

'''
wandb.init(
    project="finetunning2",
    config={
        "learning_rate":0.001,
        "architecture":"resnet50",
        "epochs": EPOCHS,
    })
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=EPOCHS)
wandb.finish()
'''

# In[16]:


# load(model_ft,dir_check='./check')
# # image_list = visualize_model(model_ft)
# save_dir = './val_image'
# if os.path.exists(save_dir)==False:
#     os.mkdir(save_dir)
# visualize_predict_save(device, dataloaders, model_ft, class_names, batch_num=len(dataloaders['val'].dataset)/_batch_size, exp_dir_path=save_dir, sel_dataset='val')


# In[ ]:





# In[17]:


# from torchsummary import summary
# print(model_ft)
# summary(model_ft, (3, 512, 512))


# In[18]:


import torch, gc
gc.collect()
torch.cuda.empty_cache()

_model=None
_device=None
img_transformer = None

from torchvision.models import ResNet50_Weights

def model_init():
    global _model, _device, img_transformer
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = _model.fc.in_features
    # 여기서 각 출력 샘플의 크기는 2로 설정합니다.
    # 또는, nn.Linear(num_ftrs, len (class_names))로 일반화할 수 있습니다.
    _model.fc = nn.Linear(num_ftrs, 2)
    _model = _model.to(_device)
    load(_model,dir_check='./check')

    img_transformer = transforms.Compose(
    [transforms.Resize((512, 512)),
    transforms.ToTensor()]
    )



def predict(img):
    global _model, img_transformer
    input = img_transformer(img).to(_device)
    input = torch.unsqueeze(input, 0)

    with torch.no_grad():
        output = _model(input)
    return output.argmax().item()
        
if __name__ == '__main__':
    model_init()
    result = predict(Image.open('/home/lkj004124/data2/spacing/test/images/20190207_314361549530754003.jpeg'))
    if result:
        print('파손')
    else: 
        print('정상')