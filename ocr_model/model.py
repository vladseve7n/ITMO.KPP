from itertools import groupby
import fastwer
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dataloader import NumberplatesDataset


num_classes = 23 # 10 digits, 12 letters, 1 blank symbol
blank_label = 22
gru_hidden_size = 128
gru_num_layers = 2
cnn_output_height = 4
cnn_output_width = 32
BATCH_SIZE = 128

TOKEN2SYMBOLS_MAPPING = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 
        10: 'A', 11: 'B', 12: 'E', 13: 'K', 14: 'M', 15: 'H', 16: 'O', 17: 'P',
         18: 'C', 19: 'T', 20: 'Y', 21: 'X'
    }


class OCR_CRNN(pl.LightningModule):
	def __init__(self, data_dir):
		super().__init__()
		self.data_dir = data_dir
		self.cnn = nn.Sequential(
				nn.Conv2d(3, 32, kernel_size=(3, 3), stride=2),
				nn.InstanceNorm2d(32),
				nn.LeakyReLU(),
				nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2),
				nn.InstanceNorm2d(32),
				nn.LeakyReLU(),
				nn.Conv2d(32, 64, kernel_size=(3, 3)),
				nn.InstanceNorm2d(64),
				nn.LeakyReLU(),
				nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2),
				nn.InstanceNorm2d(64),
				nn.LeakyReLU(),
		)
		self.gru_input_size = cnn_output_height * 64
		self.gru = nn.GRU(self.gru_input_size, gru_hidden_size, gru_num_layers, 
                          batch_first=True, bidirectional=True)
		self.fc = nn.Linear(gru_hidden_size * 2, num_classes)

		self.criterion = nn.CTCLoss(blank=blank_label, reduction='mean', zero_infinity=True)

	def forward(self, x):
		batch_size = x.shape[0]
		out = self.cnn(x)
		out = out.reshape(batch_size, -1, self.gru_input_size)
		out, _ = self.gru(out)
		out = torch.stack([F.log_softmax(self.fc(out[i]), dim=-1) for i in range(out.shape[0])])
		return out

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x_train, y_train, target_lengths = train_batch
		# mask = y_train.ne(-1)
		# nonpadded_symbols = torch.sum(mask, axis = 1)
		# print(nonpadded_symbols)
		# y_train_unpadded = []
		# for i, number_plate in enumerate(y_train):
		# 	y_train_unpadded.append(number_plate[:nonpadded_symbols[i]])
		# y_train = torch.Tensor(y_train_unpadded)
		batch_size = x_train.shape[0]  # x_train.shape == torch.Size([64, 28, 140])
		#x_train = x_train.view(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
		y_pred = self(x_train)
		y_pred = y_pred.permute(1, 0, 2)  # y_pred.shape == torch.Size([64, 32, 11]
		input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
		#target_lengths = torch.IntTensor([len(t) for t in y_train])
		loss = self.criterion(y_pred, y_train, input_lengths, target_lengths)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x_val, y_val, target_lengths = val_batch
		batch_size = x_val.shape[0]
		#x_val = x_val.view(x_val.shape[0], 1, x_val.shape[1], x_val.shape[2])
		y_pred = self(x_val)
		y_pred = y_pred.permute(1, 0, 2)
		input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
		#target_lengths = torch.IntTensor([len(t) for t in y_val])
		loss = self.criterion(y_pred, y_val, input_lengths, target_lengths)
		_, max_index = torch.max(y_pred, dim=2)
		val_correct = 0
		val_total = batch_size
		CERs = []
		for i in range(batch_size):
			raw_prediction = list(max_index[:, i].detach().cpu().numpy())
			prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label]).cuda()
			output = ''.join([TOKEN2SYMBOLS_MAPPING[token] for token in prediction.tolist() if token != blank_label])
			ref = ''.join([TOKEN2SYMBOLS_MAPPING[token] for token in y_val[i].tolist() if token != blank_label])
			CER = fastwer.score_sent(output, ref, char_level=True)
			CERs.append(CER)
			# if len(prediction) == len(y_val[i]) and torch.all(prediction.eq(y_val[i])):
			# 	val_correct += 1
		self.log('val_loss', loss)
		self.log('Mean CER per batch', sum(CERs)/batch_size)

	def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
		if stage == "fit" or stage is None:
			dataset_full = NumberplatesDataset(self.data_dir, mode='train')
			self.dataset_train, self.dataset_val = random_split(dataset_full, 
				[round(len(dataset_full) * 0.8), round(len(dataset_full) * 0.2)])
        # Assign test dataset for use in dataloader(s)
		if stage == "test" or stage is None:
			self.dataset_test = NumberplatesDataset(self.data_dir, mode='test')

	def train_dataloader(self):
		return DataLoader(self.dataset_train, batch_size=BATCH_SIZE, num_workers=12)

	def val_dataloader(self):
		return DataLoader(self.dataset_val, batch_size=64, num_workers=12)

	def test_dataloader(self):
		return DataLoader(self.dataset_test, batch_size=BATCH_SIZE, num_workers=12)