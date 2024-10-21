from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from trainer import ERCTrainer
from dataloader import MeldDataLoader
import argparse

def define_argparser():
  p = argparse.ArgumentParser()

  p.add_argument(
      '--train_dir',
      type=str,
      help='Set train file path'
  )
  p.add_argument(
      '--valid_dir',
      type=str,
      help='Set valid file path'
  )
  p.add_argument(
      '--dataset_name',
      type=str,
      default='meld',
      help='Set dataset name'
  )
  p.add_argument(
      '--batch_size',
      type=int,
      default=32,
      help='Mini batch size for gradient descent. Default=%(default)s'
  )
  p.add_argument(
      '--n_epochs',
      type=int,
      default=3,
      help='Number of epochs to train. Default=%(default)s'
  )
  p.add_argument(
      '--gpu_id',
      type=int,
      default=-1,
      help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(defalut)s'
  )
  p.add_argument(
      '--context_window',
      type=int,
      default=3,
      help='Set context window size'
  )
  p.add_argument(
      '--lr',
      type=float,
      default=5e-5,
      help='Initial learning rate. Default=%(default)s',
  )
  p.add_argument(
      '--weight_decay',
      type=float,
      default=1e-2,
      help='Initial learning rate. Default=%(default)s',
  )
  p.add_argument(
      '--lr_decay_start',
      type=int,
      default=10,
      help='Learning rate decay start at. Default=%(default)s',
  )

  p.add_argument(
      '--resume_ckpt',
      type=str,
      default=None,
      help='resume train checkpoint path',
  )

  config = p.parse_args()

  return config

def main(config):
  mlflow_logger = pl_loggers.MLFlowLogger(save_dir='logs/')

  data_loader = MeldDataLoader(train_dir=config.train_dir, valid_dir=config.valid_dir,batch_size=config.batch_size,context_window=config.context_window)

  if config.dataset_name.lower() == 'meld':
    clsNum = 7
  
  checkpoint_callback = ModelCheckpoint(
    dirpath="/content/gdrive/MyDrive/model_checkpoints/",
    filename="{epoch}-{val_loss:.2f}",  # 각 체크포인트마다 고유한 이름 생성
    save_top_k=-1,  # 모든 체크포인트 저장
    monitor="val_loss",  # validation loss를 기준으로 체크포인트 선택
    mode="min",  # val_loss가 작을수록 더 좋은 모델로 간주
    every_n_epochs=1  # 매 에포크마다 체크포인트 저장
)

  model = ERCTrainer(clsNum=clsNum, lr=config.lr, weight_decay=config.weight_decay, warmup_step=config.lr_decay_start)
  trainer = Trainer(
    max_epochs=config.n_epochs,
    devices=[config.gpu_id] if config.gpu_id >= 0 else None,
    accelerator="gpu" if config.gpu_id >= 0 else "cpu",
    enable_progress_bar=True,
    logger=mlflow_logger,
    callbacks=[checkpoint_callback],
    check_val_every_n_epoch=1
  )
  
  if config.resume_ckpt is not None:
    trainer.fit(model, data_loader, ckpt_path=config.resume_ckpt)
  else:
    trainer.fit(model, data_loader)


if __name__ == '__main__':
  config = define_argparser()
  main(config)
