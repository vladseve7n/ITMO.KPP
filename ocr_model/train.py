import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
from model import OCR_CRNN


pl.seed_everything(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


model = OCR_CRNN('/home/dmitriy/projects/ITMO.KPP/autoriaNumberplateOcrRu-2021-09-01/train/img')
trainer = pl.Trainer(accelerator="auto", gpus=1, max_epochs=50)
trainer.fit(model)
    
