import argparse

import torch.cuda

from models.melody_completion_net import MelodyCompletionNet
from dataset import MusicDataset

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v')
    parser.add_argument('--dataset', '-d', default='Maestro')

    args = parser.parse_args()

    train_dataset = MusicDataset(dataset=args.dataset, type='train', max_samples=80)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=8,
                                  num_workers=4,
                                  shuffle=True)

    val_dataset = MusicDataset(dataset=args.dataset, type='validation', max_samples=20)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=8,
                                num_workers=4,
                                shuffle=False)

    model = MelodyCompletionNet()

    logger = TensorBoardLogger('.', version=args.version)
    model_ckpt = ModelCheckpoint(dirpath=f'lightning_logs/{args.version}/checkpoints',
                                 save_top_k=0,
                                 monitor='loss_val',
                                 save_weights_only=True)
    lr_monitor = LearningRateMonitor()

    trainer = Trainer(accelerator='auto',
                      devices=1 if torch.cuda.is_available() else None,
                      max_epochs=150,
                      val_check_interval=200,
                      callbacks=[model_ckpt, lr_monitor],
                      logger=logger)
    trainer.fit(model, train_dataloader, val_dataloader)
