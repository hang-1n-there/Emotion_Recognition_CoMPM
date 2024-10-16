from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import Trainer
from dataloader import MeldDataLoader
import argparse
from trainer import ERCTrainer

def define_argparser():
  p = argparse.ArgumentParser()

  p.add_argument(
      '--train_dir',
      type=str,
      help='Set train file path'
  )
  p.add_argument(
      '--valid_dir',
      type='str',
      help='Set valid file path'
  )
  p.add_argument(
      '--dataset_name',
      type='str',
      defalut='meld',
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
      default=20,
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
      default=1.,
      help='Initial learning rate. Default=%(default)s',
  )
  p.add_argument(
      '--lr_decay_start',
      type=int,
      default=10,
      help='Learning rate decay start at. Default=%(default)s',
  )
  config = p.parse_args()

  return config

def main(config):
  mlflow_logger = pl_loggers.MLFlowLogger(save_dir='logs/')

  data_loader = MeldDataLoader(train_dir=config.train_dir, valid_dir=config.valid_dir,batch_size=config.batch_size,context_window=config.context_window)

  if config.dataset_name.lower() == 'meld':
    clsNum = 7

  model = ERCTrainer(clsNum=clsNum, lr=config.lr, weight_decay=config.weight_decay, warmup_step=config.warmup_step)
  trainer = Trainer(max_epochs=config.n_epochs, gpus=config.gpu_id, progress_bar_refresh_rate=20, logger=mlflow_logger)

  trainer.fit(model, data_loader)


if __name__ == '__main__':
  config = define_argparser()
  main(config)
