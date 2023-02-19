# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
import pandas as pd
import cv2
from mylog import make_logger ,print_log
from skimage.measure import label, regionprops, find_contours
import scipy.ndimage as ndimage
import random
import wandb
wandb.login()

# Set Current Working Directory as Script Running Directory
def set_cwd_as_srd():
    path_cur_script_file = os.path.abspath(__file__)
    path_cur_dir= os.path.dirname(path_cur_script_file)
    os.chdir(path_cur_dir)

def make_exp_dir():
    # 1. exp로 시작하는 디렉토리 생성
    dir_exp_list = [ f.name for f in os.scandir('.') if f.is_dir() and f.name.startswith('exp') ]
    if len(dir_exp_list)==0:
        os.mkdir('./exp1')
        return './exp1'
    else: # exp로 시작하는 디렉토리가 존재하는 경우
        dir_exp_list.sort(key=lambda x:int(x[3:]),reverse=True) # as decending
        exp_dir_name = './exp'+str(int(dir_exp_list[0][3:])+1) #exp+(이전마지막번호+1) 로 디렉토리 생성
        os.mkdir(exp_dir_name)
        return exp_dir_name

def get_lastest_exp_dir():
    dir_exp_list = [ f.name for f in os.scandir('.') if f.is_dir() and f.name.startswith('exp') ]
    if len(dir_exp_list)==0:
       return None 
    else: # exp로 시작하는 디렉토리가 존재하는 경우
        dir_exp_list.sort(key=lambda x:int(x[3:]),reverse=True) # as decending
        exp_dir_name = './exp'+str(int(dir_exp_list[0][3:])) #exp+(이전마지막번호) 
        return exp_dir_name 

# class TqdmToLoggerHandler(logging.Handler):
#     def __init__(self, level=logging.NOTSET):
#         super().__init__(level=level)

#     def emit(self, record):
#         try:
#             msg = self.format(record=record)
#             tqdm.write(msg)
#             self.flush()
#         except Exception:
#             self.handleError(record=record)



def save(netG, optimG, epoch, dir_weight='.'):
    dir_check = os.path.join(dir_weight,'weight')
    if not os.path.exists(dir_check):
        os.makedirs(dir_check)

    torch.save({'netG': netG.state_dict(),
                'optimG': optimG.state_dict()},
                '%s/model_epoch%04d.pth' % (dir_check, epoch))
    print_log(f'save {dir_check}/model_epoch{epoch:04d}.pth')

def load(device, netG, optimG=None, epoch=None,dir_weight='.'):
    dir_check = os.path.join(dir_weight,'weight')
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

    print_log(f'load {dir_check}/model_epoch{epoch:04d}.pth')

    if optimG is None:
        netG.load_state_dict(dict_net['netG'])
        return netG, epoch
    else:
        netG.load_state_dict(dict_net['netG'])
        optimG.load_state_dict(dict_net['optimG'])
        return netG, optimG, epoch

""" Convert a mask to border image """
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

""" Mask to bounding boxes, polygons """
def mask_to_bbox_polygon(mask):
    bboxes = []
    polygons=[]

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])
        polygons.append(prop.coords)

    return bboxes,polygons

class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# # SegDataSet 생성
class MySegDataset(object):
    """socar (spacing,dent,scratch) image dataset."""

    def __init__(self, csv_file, root_dir, transform=None, small_remove=False, normal_remove=False, random_rotate=False):
        """
        Args:
            csv_file (string): csv 파일의 경로
            root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.random_rotate = random_rotate
        self.normal_remove = normal_remove
        self.rand_rot_idx=0
        self.rand_rot={0:45, 1:90+45,2:180+45,3:270+45,
                4:Image.FLIP_LEFT_RIGHT,5:Image.ROTATE_90,
                6:Image.FLIP_TOP_BOTTOM,7:Image.ROTATE_180,8:Image.ROTATE_270}
        self.classtype_to_idx = {'background':0,'spacing':1,'dent':2,'scratch':3}
        self.idx_to_classtype = dict((value, key) for (key, value) in self.classtype_to_idx.items())
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transform
        self.small_remove = small_remove
        if self.small_remove==True or self.normal_remove == True :
            self.remove_small_data()

    def make_random_idx(self):
       self.rand_rot_idx = random.randint(0,9) 

    def get_random_rotate(self,img):
        if self.random_rotate == True:
            if self.rand_rot_idx < 4: 
                img = img.rotate(self.rand_rot[self.rand_rot_idx],expand=True)
            elif self.rand_rot_idx <9:
                img = img.transpose(self.rand_rot[self.rand_rot_idx])

        return img
    def __len__(self):
        return len(self.data_frame)

    def remove_small_data(self):
        # small data를 제거한 후에 객체가 존재하지 않는 경우 dataframe에서 제거시킴
        drop_index = []
        for idx in tqdm(range(len(self.data_frame))):
            row = self.data_frame.iloc[idx]

            inst_bbox_list=[]
            if row.label==True:#mask exist
                for (class_type,is_true) in zip(row.index[1:4].to_list(), row[1:4].to_list()):
                    if (is_true==True):
                        msk_path = os.path.join(self.root_dir,class_type)
                        msk_path = os.path.join(msk_path,row.prefix)
                        img_path = msk_path
                        msk_path = os.path.join(msk_path,'masks')
                        msk_path = os.path.join(msk_path,row.filename)
                        img_path = os.path.join(img_path,'images')
                        img_path = os.path.join(img_path,row.filename)

                        mask= Image.open(msk_path)
                        mask_arr = np.array(mask)
                        mask_gray= cv2.cvtColor(mask_arr,cv2.COLOR_RGB2GRAY)
                        bboxes,polygons = mask_to_bbox_polygon(mask_gray)
                        
                        for i in range(len(bboxes)):
                            inst_bbox_list.append(bboxes[i])
                            
                if (self.small_remove==True) and (len(inst_bbox_list)>1):
                    remove_index=[]
                    for i in range(len(inst_bbox_list)):
                        box = inst_bbox_list[i]
                        area = (box[3] - box[1]) * (box[2] - box[0])
                        if int(np.sqrt(area)) <= 10: # 10x10 보다 작거나 같은것은 제외시킨다
                            remove_index.append(i)
                    
                    remove_index.sort(reverse=True) # 큰값부터지우기위해
                    #print_log(f'remove_index={remove_index}')
                    for i in range(len(remove_index)):
                        inst_bbox_list.pop(remove_index[i])

                if len(inst_bbox_list)==0:
                    drop_index.append(idx)
            else:# mask is not exist
                if self.normal_remove==True:
                    drop_index.append(idx)
        
        self.data_frame.drop(index=drop_index,inplace=True)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        self.make_random_idx()
        # each polygon area to instance mask
        inst_mask_list=[]
        inst_bbox_list=[]
        inst_label_list=[]
        if row.label==True:#mask exist
            for (class_type,is_true) in zip(row.index[1:4].to_list(), row[1:4].to_list()):
                if (is_true==True):
                    msk_path = os.path.join(self.root_dir,class_type)
                    msk_path = os.path.join(msk_path,row.prefix)
                    img_path = msk_path
                    msk_path = os.path.join(msk_path,'masks')
                    msk_path = os.path.join(msk_path,row.filename)
                    img_path = os.path.join(img_path,'images')
                    img_path = os.path.join(img_path,row.filename)

                    mask= Image.open(msk_path)
                    mask = self.get_random_rotate(mask)
                    mask_arr = np.array(mask)
                    mask_gray= cv2.cvtColor(mask_arr,cv2.COLOR_RGB2GRAY)
                    bboxes,polygons = mask_to_bbox_polygon(mask_gray)
                    
                    for i in range(len(bboxes)):
                        tmp_msk = np.zeros_like(mask_gray,dtype=np.uint8)
                        pg = polygons[i].copy()
                        tmp_msk[np.round(pg[:,0]).astype('int'),np.round(pg[:,1]).astype('int')]=1
                        tmp_msk = ndimage.binary_fill_holes(tmp_msk)
                        #pg[:,[0, 1]] = pg[:,[1, 0]]
                        #cv2.fillPoly(tmp_msk, pts=[pg], color=(1,0,0))
                        inst_mask_list.append(tmp_msk)
                        inst_bbox_list.append(bboxes[i])
                        inst_label_list.append(self.classtype_to_idx[class_type])
                        
            img = Image.open(img_path).convert("RGB")
            img = self.get_random_rotate(img)
            if (self.small_remove==True) and (len(inst_bbox_list)>1):
                remove_index=[]
                for i in range(len(inst_bbox_list)):
                    box = inst_bbox_list[i]
                    area = (box[3] - box[1]) * (box[2] - box[0])
                    if int(np.sqrt(area)) <= 10: # 10x10 보다 작거나 같은것은 제외시킨다
                        remove_index.append(i)
                
                remove_index.sort(reverse=True) # 큰값부터지우기위해
                #print_log(f'remove_index={remove_index}')
                for i in range(len(remove_index)):
                    inst_bbox_list.pop(remove_index[i])
                    inst_mask_list.pop(remove_index[i])
                    inst_label_list.pop(remove_index[i])
            
            inst_mask = np.stack(inst_mask_list,axis=0)
            # get bounding box coordinates for each mask
            num_objs = len(inst_bbox_list)

            boxes = torch.as_tensor(inst_bbox_list, dtype=torch.float32)
            # there is only one class
            labels= torch.as_tensor(inst_label_list, dtype=torch.int64)
            masks = torch.as_tensor(inst_mask, dtype=torch.uint8)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)


        else:#row.label=False mask is not exist
            for i in range(1,len(self.idx_to_classtype)):
                img_path = os.path.join(self.root_dir,self.idx_to_classtype[i])
                img_path = os.path.join(img_path,row.prefix)
                img_path = os.path.join(img_path,'images')
                img_path = os.path.join(img_path,row.filename)
                if os.path.exists(img_path)==True:
                    break
            img = Image.open(img_path).convert("RGB")
            img = self.get_random_rotate(img)
            num_objs = len(inst_bbox_list)
            boxes = torch.as_tensor(inst_bbox_list, dtype=torch.float32)
            # there is only one class
            labels= torch.as_tensor(inst_label_list, dtype=torch.int64)
            masks = torch.as_tensor([], dtype=torch.uint8)

            image_id = torch.tensor([idx])
            area = torch.zeros((num_objs,), dtype=torch.int64)
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target



def get_model_instance_segmentation(num_classes,pretrained=True):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    #if train:
        #transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def print_metric(arr):
    prt_fmt = f'''
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {arr[0]:.3f}
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {arr[1]:.3f}
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {arr[2]:.3f}
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {arr[3]:.3f}
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {arr[4]:.3f}
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {arr[5]:.3f}
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {arr[6]:.3f}
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {arr[7]:.3f}
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {arr[8]:.3f}
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {arr[9]:.3f}
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {arr[10]:.3f}
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {arr[11]:.3f}
        '''
    print_log(prt_fmt)


def run_train():
    exp_dir_path = make_exp_dir()
    make_logger(exp_dir_path)
    print_log(f'exp_dir={exp_dir_path}')

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # use our dataset and defined transformations
    data_folder = '../dataset1'
    dataset = MySegDataset(os.path.join(data_folder,'socar_train_info_segment_all.csv'),data_folder,get_transform(train=True),small_remove=True,normal_remove=True,random_rotate=True)
    dataset_test = MySegDataset(os.path.join(data_folder,'socar_test_info_segment_all.csv'),data_folder,get_transform(train=False),small_remove=True,normal_remove=True,random_rotate=False)
   
    # our dataset has two classes only - background and person
    num_classes = len(dataset.idx_to_classtype)
    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes,pretrained=True)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005,
    #                             momentum=0.9, weight_decay=0.0005)
    # # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.1)
    optimizer = torch.optim.Adam(params, lr=0.0001)
    '''
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                             lr_lambda=lambda epoch: 0.95 ** epoch,
                                             last_epoch=-1,
                                             verbose=False)
    '''
    # let's train it for 10 epochs
    num_epochs = 50

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_metric = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, exp_dir_path=exp_dir_path)
        print_log(train_metric)
        
        # update the learning rate
        #lr_scheduler.step()
        
        # evaluate on the test dataset
        tmp_eval = evaluate(model, data_loader_test, device=device)
        
        for tmp_iou_type, tmp_coco_eval in tmp_eval.coco_eval.items():
             print_log("IoU metric: {}".format(tmp_iou_type))
             print_metric(tmp_coco_eval.stats)

        save(model, optimizer, epoch, exp_dir_path)


import matplotlib.pyplot as plt
import random
import cv2
def random_colour_masks(image, label_idx):
    idx_color = {1:[255,0,0],2:[0,255,0],3:[0,0,255]} # spacing, dent, scratch
    im_r = np.zeros_like(image).astype(np.uint8)
    im_g = np.zeros_like(image).astype(np.uint8)
    im_b = np.zeros_like(image).astype(np.uint8)
    #r = random.randint(0,255)
    #g = random.randint(0,255)
    #b = random.randint(0,255)
    r = idx_color[label_idx][0]
    g = idx_color[label_idx][1]
    b = idx_color[label_idx][2]
    im_r[image == 1] =r
    im_g[image == 1] =g
    im_b[image == 1] =b
    coloured_mask = np.stack([im_r, im_g, im_b], axis=2)
    return coloured_mask,r,g,b

import gc
def imshow_result(img_org, gt_bboxes, gt_masks, gt_labels, pred_bboxes, pred_masks, pred_labels, save_file=None):
    plt.close('all')
    plt.clf()
    col = 3
    row = 1
    fig = plt.figure(figsize=(int(9*col),int(9*row)))

    inp = img_org.transpose((1, 2, 0))
    inp = inp * 255 # 0~1 to 255
    inp = np.array(inp,dtype=np.uint8)
    # 정답
    im2 = inp.copy()
    area_list = []
    for i in range(len(gt_bboxes)):
        rec=gt_bboxes[i].astype(dtype=np.uint32)
        '''
        msk=gt_masks[i]
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        im2[:, :, 0][msk == 1] = r
        im2[:, :, 1][msk == 1] = g
        im2[:, :, 2][msk == 1] = b
        '''
        rgb_msk,r,g,b = random_colour_masks(gt_masks[i],gt_labels[i])
        im2 = cv2.addWeighted(im2,1,rgb_msk,0.5,0)
        cv2.rectangle(im2, (rec[0],rec[1]),(rec[2],rec[3]),(r,g,b),2)
        #cv2.putText(im2,gt_labels[i], (rec[0],rec[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (b,g,r),thickness=2)


    # 예측
    im3 = inp.copy()
    for i in range(len(pred_bboxes)):
        rec=pred_bboxes[i]
        msk=pred_masks[i]
       
        
        if im3.shape[:2] != msk.shape:
        #if im3.shape[0] != msk.shape[0] or im3.shape[1] != msk.shape[1] :
            print_log(f'{i} im3.shape={im3.shape}, msk.shape={msk.shape}')
            continue
        '''
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        im3[:, :, 0][msk == 1] = r
        im3[:, :, 1][msk == 1] = g
        im3[:, :, 2][msk == 1] = b
        ''' 
        rgb_msk,r,g,b = random_colour_masks(pred_masks[i],pred_labels[i])
        im3 = cv2.addWeighted(im3,1,rgb_msk,0.5,0)
        cv2.rectangle(im3, (rec[0],rec[1]),(rec[2],rec[3]),(r,g,b),2)
        #cv2.putText(im3,pred_labels[i], (rec[0],rec[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (b,g,r),thickness=2)

    ax = fig.add_subplot(row,col,1)
    ax.set_axis_off()
    ax.set_title('org')
    ax.imshow(inp)
    ax2 = fig.add_subplot(row,col,2)
    ax2.set_axis_off()
    ax2.set_title('y_seg_num= '+str(len(gt_bboxes)))
    ax2.imshow(im2)
    ax3 = fig.add_subplot(row,col,3)
    ax3.set_axis_off()
    ax3.set_title('pred_seg_num= '+str(len(pred_bboxes)))
    ax3.imshow(im3)

    if save_file is not None:
        plt.savefig(save_file)
        plt.close('all')
        plt.clf()
    else:
        plt.show()


def predict(device, model, img, gt, threshold, save_file):
    model.eval()
    pred = model([img.to(device)])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold]
    pred_t = len(pred_t)
    pred_masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    pred_class = pred[0]['labels'].detach().cpu().numpy()
    pred_boxes = [[int(i[0]), int(i[1]), int(i[2]), int(i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_masks = pred_masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    imshow_result(img.cpu().numpy(), \
                gt['boxes'].cpu().numpy(), \
                gt['masks'].detach().cpu().numpy(), \
                gt['labels'].detach().cpu().numpy(), \
                pred_boxes,pred_masks,pred_class, \
                save_file=save_file)

from tqdm import tqdm
def run_eval(threshold=0.5):
    exp_dir_path = get_lastest_exp_dir()
    if exp_dir_path==None:
        print('there is not exp# folder, so quit run_eval')
        return
    make_logger(exp_dir_path)
    save_eval_dir = os.path.join(exp_dir_path,'eval')
    if os.path.exists(save_eval_dir)==False:
        os.mkdir(save_eval_dir)
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # use our dataset and defined transformations
    data_folder = '../dataset1'
    #dataset = MySegDataset(os.path.join(data_folder,'socar_train_info_segment_all.csv'),data_folder,get_transform(train=True),small_remove=False)
    dataset_test = MySegDataset(os.path.join(data_folder,'socar_test_info_segment_all.csv'),data_folder,get_transform(train=False),small_remove=False,normal_remove=True,random_rotate=False)
   
    # our dataset has two classes only - background and person
    num_classes = len(dataset_test.idx_to_classtype)
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes,pretrained=False)
    load(device, model,dir_weight=exp_dir_path)

    # move model to the right device
    model.to(device)
    my_class_name = dataset_test.idx_to_classtype

    with torch.no_grad():
        for i in tqdm(range(len(dataset_test))):
            # pick one image from the test set
            img, gt = dataset_test[i]
            # put the model in evaluation mode
            predict(device, model, img, gt, threshold, os.path.join(save_eval_dir,str(i)+'.png'))
            gc.collect()

    # we need to convert the image, which has been rescaled to 0-1 and 
    # had the channels flipped so that we have it in [C, H, W] format.
        #Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()).save(os.path.join(save_eval_dir,str(i)+'_org.png'))

    # visualize the top predicted segmentation mask. 
    # The masks are predicted as [N, 1, H, W], 
    # where N is the number of predictions, and are probability maps between 0-1.
        #Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy()).save(os.path.join(save_eval_dir,str(i)+'_msk.png'))

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

if __name__ == "__main__":
    set_cwd_as_srd()
    set_seed(42)
    #run_train()
    run_eval()
