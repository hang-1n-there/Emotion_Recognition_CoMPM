import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model import ERC_model

class ERCTrainer(pl.LightningModule):
  def __init__(self, clsNum, lr=5e-5, weight_decay=1e-2, warmup_step=32):
    super(ERCTrainer, self).__init__()
    self.model = ERC_model(clsNum)
    self.crit = nn.CrossEntropyLoss()
    self.lr = lr
    self.weight_decay = weight_decay
    self.warmup_steps = warmup_step
  def forward(self, batch_padding_token, batch_padding_attention_mask, batch_PM_input):
    return self.model(batch_padding_token, batch_padding_attention_mask, batch_PM_input)
  
  # batch : collate fn return
  def training_step(self, batch, batch_idx):
    batch_padding_token, batch_padding_attention_mask, batch_PM_input, labels = batch

    # self : forward
    logits = self(batch_padding_token, batch_padding_attention_mask, batch_PM_input)

    loss = self.crit(logits, labels)
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss
  
  def vallidation_step(self, batch, batch_idx):
    batch_padding_token, batch_padding_attention_mask, batch_PM_input, labels = batch

    # self : forward
    logits = self(batch_padding_token, batch_padding_attention_mask, batch_PM_input)
    
    loss = self.crit(logits, labels)
    preds = torch.argmax(logits, dim=1)
    acc = (preds == labels).float().mean()

    self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True,logger=True)
    self.log('val_acc', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss
  
  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=self.warmup_steps, num_training_step=self.trainer.estimated_stepping_batches
    )
    return [optimizer], [scheduler]