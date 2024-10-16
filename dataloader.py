import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from transformers import RobertaModel
from torch.utils.data import DataLoader, Dataset
from dataset import MeldDataset

class MeldDataLoader(pl.LightningDataModule):
  def __init__(self, train_dir: str, valid_dir: str, batch_size: int = 32, context_window=None):
    super().__init__()
    self.train_dir = train_dir
    self.valid_dir = valid_dir
    self.batch_size = batch_size
    self.context_window = context_window

  def setup(self, stage=None):
    self.train_dataset = MeldDataset(self.train_dir, context_window=self.context_window)
    self.val_dataset = MeldDataset(self.valid_dir, context_window=self.context_window)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=self.train_dataset.collate_fn)

  def valid_dataloader(self):
    return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=self.val_dataset.collate_fn)
