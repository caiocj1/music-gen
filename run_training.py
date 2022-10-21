import argparse

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

    train_dataset = MusicDataset(dataset=args.dataset, type='train', max_samples=20)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=10,
                                  num_workers=0,
                                  shuffle=True)

    val_dataset = MusicDataset(dataset=args.dataset, type='validation', max_samples=2)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=10,
                                num_workers=0,
                                shuffle=False)

    model = MelodyCompletionNet()

    logger = TensorBoardLogger('.', version=args.version)
    model_ckpt = ModelCheckpoint(dirpath=f'lightning_logs/{args.version}/checkpoints',
                                 save_top_k=1,
                                 monitor='loss_val')
    lr_monitor = LearningRateMonitor()

    trainer = Trainer(accelerator='gpu',
                      devices=1,
                      max_epochs=150,
                      val_check_interval=40,
                      callbacks=[model_ckpt, lr_monitor],
                      logger=logger)
    trainer.fit(model, train_dataloader, val_dataloader)
