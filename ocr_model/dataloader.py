import os
import numpy as np
import cv2

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
    '''
    def __init__(
        self, 
        images_path, 
        mode='train'):
        self.mode = mode
        self.images_path = images_path
        self.filenames = os.listdir(self.images_path)
        self.image_transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Resize((64, 256)),
                                torchvision.transforms.Normalize(
                                    mean=[0.5] * 3,
                                    std=[0.5] * 3)
                            ])
        self.max_target_len = 11


    def __getitem__(self, i):
        filename = self.filenames[i]
        target_tensor = PLATE_SYMBOLS_MAPPING['pad_token']*torch.ones(self.max_target_len)
        plate_number = filename.replace('.png','')
        if '_' in plate_number:
            plate_number = plate_number[:-2]
        plate_number = torch.Tensor([PLATE_SYMBOLS_MAPPING[sym] for sym in plate_number])
        target_tensor[:plate_number.shape[0]] = plate_number
        plate_num_img = cv2.imread(f'{self.images_path}/{filename}')
        plate_num_img = cv2.cvtColor(plate_num_img, cv2.COLOR_BGR2RGB) 
        plate_num_img = self.image_transform(plate_num_img)
        # pad plate number
        return (
            plate_num_img,
            target_tensor
        )

    def __len__(self):
        return len(self.filenames)