import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import OCR_CRNN


pl.seed_everything(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


model = OCR_CRNN('../autoriaNumberplateOcrRu-2021-09-01/train/img', 
    padding_type='reflect_padding')
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="Accuracy",
    mode="max")
logger = TensorBoardLogger("lightning_logs", name="pretrained_resnet18_2GRU_multistep_lr_reflectpad_newsteps")
trainer = pl.Trainer(accelerator="auto", gpus=1,
    max_epochs=55, logger=logger, callbacks=[checkpoint_callback])
trainer.fit(model)
