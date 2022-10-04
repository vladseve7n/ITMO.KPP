from itertools import groupby
import fastwer
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from ocr_model.dataloader import NumberplatesDataset
from torchvision import models


num_classes = 23 # 10 digits, 12 letters, 1 blank symbol
blank_label = 22
gru_hidden_size = 128
cnn_output_width = 32
BATCH_SIZE = 128

TOKEN2SYMBOLS_MAPPING = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 
        10: 'A', 11: 'B', 12: 'E', 13: 'K', 14: 'M', 15: 'H', 16: 'O', 17: 'P',
         18: 'C', 19: 'T', 20: 'Y', 21: 'X'
    }


class OCR_CRNN(pl.LightningModule):
	def __init__(self, data_dir, gru_num_layers=2, padding_type='resize'):
		super().__init__()
		self.data_dir = data_dir
		self.padding_type = padding_type
		self.res_net = models.resnet18(pretrained=True)
		self.gru_input_size = 64
		self.fc_1 = nn.Linear(1000, 2048)
		
		self.gru = nn.GRU(self.gru_input_size, gru_hidden_size, gru_num_layers, 
                          batch_first=True, bidirectional=True)
		self.fc = nn.Linear(gru_hidden_size * 2, num_classes)
		self.criterion = nn.CTCLoss(blank=blank_label, reduction='mean', zero_infinity=True)
		self.save_hyperparameters(ignore=['data_dir'])

	def forward(self, x):
		batch_size = x.shape[0]
		out = self.res_net(x)
		out = self.fc_1(out)
		out = out.reshape(batch_size, -1, self.gru_input_size)
		out, _ = self.gru(out)
		out = torch.stack([F.log_softmax(self.fc(out[i]), dim=-1) for i in range(out.shape[0])])
		return out

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		scheduler = MultiStepLR(optimizer, milestones=[27,32,36,40,43,46], gamma=0.5) # was [30,40]
		return [optimizer], [scheduler]

	def training_step(self, train_batch, batch_idx):
		x_train, y_train, target_lengths = train_batch
		batch_size = x_train.shape[0]
		y_pred = self(x_train)
		y_pred = y_pred.permute(1, 0, 2)
		input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
		loss = self.criterion(y_pred, y_train, input_lengths, target_lengths)
		self.log('train_loss', loss)
		return loss

	def _calculate_metrics(self, y_pred, y_val, batch_size):
		_, max_index = torch.max(y_pred, dim=2)
		val_correct = 0
		CERs = []
		for i in range(batch_size):
			raw_prediction = list(max_index[:, i].detach().cpu().numpy())
			prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label]).cuda()
			output = ''.join([TOKEN2SYMBOLS_MAPPING[token] for token in prediction.tolist() if token != blank_label])
			ref = ''.join([TOKEN2SYMBOLS_MAPPING[token] for token in y_val[i].tolist() if token != blank_label])
			CER = fastwer.score_sent(output, ref, char_level=True)
			CERs.append(CER)
			if output == ref:
				val_correct += 1
		cer = sum(CERs)/batch_size
		accuracy = val_correct/batch_size
		return cer, accuracy

	def validation_step(self, val_batch, batch_idx):
		x_val, y_val, target_lengths = val_batch
		batch_size = x_val.shape[0]
		y_pred = self(x_val)
		y_pred = y_pred.permute(1, 0, 2)
		input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
		loss = self.criterion(y_pred, y_val, input_lengths, target_lengths)
		cer, accuracy = self._calculate_metrics(y_pred, y_val, batch_size)
		self.log('val_loss', loss)
		self.log('Mean CER per batch', cer)
		self.log('Accuracy', accuracy)

	def setup(self, stage=None):
        # train/val datasets for use in dataloaders
		if stage == "fit" or stage is None:
			dataset_full = NumberplatesDataset(self.data_dir,
				mode='train', padding_type=self.padding_type)
			self.dataset_train, self.dataset_val = random_split(dataset_full, 
				[round(len(dataset_full) * 0.8), round(len(dataset_full) * 0.2)])
			# self.dataset_train = NumberplatesDataset(f'{self.data_dir}/train/img',
			# 	mode='train', padding_type=self.padding_type)
			# self.dataset_val = NumberplatesDataset(f'{self.data_dir}/val/img',
			# 	mode='val', padding_type=self.padding_type)
        # test dataset for use in dataloader
		if stage == "test" or stage is None:
			self.dataset_test = NumberplatesDataset(f'{self.data_dir}/test/img',
				mode='test', padding_type=self.padding_type)

	def train_dataloader(self):
		return DataLoader(self.dataset_train, batch_size=BATCH_SIZE, num_workers=12)

	def val_dataloader(self):
		return DataLoader(self.dataset_val, batch_size=64, num_workers=12)

	def test_dataloader(self):
		return DataLoader(self.dataset_test, batch_size=BATCH_SIZE, num_workers=12)