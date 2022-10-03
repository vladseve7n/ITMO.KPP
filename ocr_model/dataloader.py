import os
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


PLATE_SYMBOLS_MAPPING = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
    'A':10,'B':11,'E':12,'K':13,'M':14,'H':15,'O':16,'P':17,'C':18,'T':19,'Y':20,'X':21,
    'pad_token':22
    }


class NumberplatesDataset(Dataset):
    '''
    mode: 'train', 'test', 'val'
    padding_type: 'resize', 'const_padding', 'reflect_padding'
    '''
    def __init__(
        self, 
        images_path, padding_type='resize',
        mode='train'):
        self.mode = mode
        self.padding_type = padding_type
        self.images_path = images_path
        self.filenames = os.listdir(self.images_path)
        self.const_padding_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((56, 224)),
            torchvision.transforms.Pad((0,84), padding_mode='constant') # pad to 224x224
        ])
        self.reflect_padding_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((56, 224)),
            torchvision.transforms.Pad((0,56), padding_mode='reflect') # pad to 56*3x224
        ])
        self.image_transform = torchvision.transforms.Compose([
            # torchvision.transforms.functional.rgb_to_grayscale(num_output_channels=3),
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])
        self.max_target_len = 11


    def __getitem__(self, i):
        filename = self.filenames[i]
        target_tensor = PLATE_SYMBOLS_MAPPING['pad_token']*torch.ones(self.max_target_len)
        plate_number = filename.replace('.png','')
        if '_' in plate_number:
            plate_number = plate_number.split('_')[0]
        plate_number = torch.Tensor([PLATE_SYMBOLS_MAPPING[sym] for sym in plate_number])
        plate_num_len = plate_number.shape[0]
        target_tensor[:plate_num_len] = plate_number
        plate_num_img = Image.open(f'{self.images_path}/{filename}')
        if self.padding_type == 'const_padding':
            plate_num_img = self.const_padding_transform(plate_num_img)
        elif self.padding_type == 'reflect_padding':
            plate_num_img = self.reflect_padding_transform(plate_num_img)
        plate_num_img = self.image_transform(plate_num_img)
        return (
            plate_num_img,
            target_tensor,
            plate_num_len
        )

    def __len__(self):
        return len(self.filenames)