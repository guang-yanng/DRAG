import os, sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import json
import random
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])

full_transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            normalize])

class IPDataset_FromFolder(data.Dataset):
    def __init__(self, image_dir, full_im_transform = None):
        super(IPDataset_FromFolder, self).__init__()

        self.image_dir = image_dir
        self.full_im_transform = full_im_transform
        
        private_imgs = os.listdir(image_dir + 'private/')
        private_imgs = [image_dir + 'private/' + img for img in private_imgs]

        public_imgs = os.listdir(image_dir + 'public/')
        public_imgs = [image_dir + 'public/' + img for img in public_imgs]
       
        self.imgs = private_imgs + public_imgs
        self.labels = [0] * len(private_imgs) + [1] * len(public_imgs)



    def __getitem__(self, index):

        img = Image.open(self.imgs[index]).convert('RGB') # convert gray to rgb
        
        target = self.labels[index]

        if self.full_im_transform:
            full_im = self.full_im_transform(img)
        else:
            full_im = img

        return target, full_im, bboxes_14, categories

    def __len__(self):
        return len(self.imgs)

    
    
    
class IPDataset_FromFileList(data.Dataset):
    def __init__(self, data_file, full_im_transform = None):
        super(IPDataset_FromFileList, self).__init__()

        self.data_file = data_file
        self.full_im_transform = full_im_transform

        data_list = pd.read_csv(data_file)
       
        self.imgs = data_list['img_name'].values.tolist()
        self.labels = data_list['label'].values.tolist()
        self.labels = [0 if label == 'private' else 1 for label in self.labels]


    def __getitem__(self, index):

        img = Image.open(self.imgs[index]).convert('RGB') # convert gray to rgb
        
        target = self.labels[index]

        if self.full_im_transform:
            full_im = self.full_im_transform(img)
        else:
            full_im = img

        return target, full_im

    def __len__(self):
        return len(self.imgs)