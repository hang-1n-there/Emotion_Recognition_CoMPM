import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torchmetrics import F1Score

from model import ERC_model

class ERCTrainer(pl.LightningModule):
    def __init__(self, clsNum, lr=5e-5, weight_decay=1e-2, warmup_step=32):
        super(ERCTrainer, self).__init__()
        self.model = ERC_model(clsNum)
        self.crit = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_step

        self.f1_macro = F1Score(task='multiclass', num_classes=clsNum, average='macro')
        self.validation_outputs = []  # Validation 결과를 저장할 리스트

    def forward(self, batch_padding_token, batch_padding_attention_mask, batch_PM_input):
        return self.model(batch_padding_token, batch_padding_attention_mask, batch_PM_input)

    # batch : collate fn return
    def training_step(self, batch, batch_idx):
        batch_padding_token, batch_padding_attention_mask, batch_PM_input, labels = batch

        # self : forward
        logits = self(batch_padding_token, batch_padding_attention_mask, batch_PM_input)

        loss = self.crit(logits, labels)
        # 명시적으로 batch_size 전달
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_padding_token.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        batch_padding_token, batch_padding_attention_mask, batch_PM_input, labels = batch

        # self : forward
        logits = self(batch_padding_token, batch_padding_attention_mask, batch_PM_input)

        loss = self.crit(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # F1 Score 업데이트 및 Validation 결과 저장
        self.f1_macro.update(preds, labels)
        self.validation_outputs.append(loss)

        # 명시적으로 batch_size 전달
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_padding_token.size(0))
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_padding_token.size(0))
        return loss

    def on_validation_epoch_end(self):
        # Validation epoch이 끝난 후 F1 Score 계산 및 로그
        macro_f1 = self.f1_macro.compute()
        self.log('val_macro_f1', macro_f1, prog_bar=True)

        # F1 Score 및 validation outputs 초기화
        self.f1_macro.reset()
        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]
