import torch.optim
from pytorch_lightning import LightningModule
import torch.nn as nn
from models.nn_blocks import CompletionNetwork, LocalDiscriminator, GlobalDiscriminator

class MelodyCompletionNet(LightningModule):

    def __init__(self):
        super(MelodyCompletionNet, self).__init__()

        self.completion_net = CompletionNetwork()

        self.local_discriminator = LocalDiscriminator()

        self.global_discriminator = GlobalDiscriminator()

        self.discriminate = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def training_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        self.log_metrics(metrics, 'train')
        self.log('loss_train', loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        self.log_metrics(metrics, 'val')
        self.log('loss_val', loss, on_step=False, on_epoch=True, logger=True)

        return loss

    def _shared_step(self, batch):
        gen_is_real_prob, real_is_real_prob, completed_img = self.forward(batch)

        loss = self.calc_loss(batch, gen_is_real_prob, real_is_real_prob, completed_img)

        metrics = self.calc_metrics()

        return loss, metrics

    def forward(self, batch):
        masked_input = torch.stack((batch['measure_img'] * (1 - batch['mask']), batch['mask']), dim=1).float()
        completed_img = self.completion_net(masked_input)

        batch_size = len(completed_img)

        gen_local_img = completed_img[batch['mask'][:, None].bool()].reshape(batch_size, 1, 128, 100)
        gen_local_vec = self.local_discriminator(gen_local_img)
        gen_global_vec = self.global_discriminator(completed_img)
        gen_discr_vec = torch.cat((gen_local_vec, gen_global_vec), dim=1)
        gen_is_real_prob = self.discriminate(gen_discr_vec)[:, 0]

        real_local_img = batch['measure_img'][batch['mask'].bool()].reshape(batch_size, 1, 128, 100).float()
        real_local_vec = self.local_discriminator(real_local_img)
        real_global_vec = self.global_discriminator(batch['measure_img'][:, None].float())
        real_discr_vec = torch.cat((real_local_vec, real_global_vec), dim=1)
        real_is_real_prob = self.discriminate(real_discr_vec)[:, 0]

        return gen_is_real_prob, real_is_real_prob, completed_img

    def calc_loss(self, batch, gen_is_real_prob, real_is_real_prob, completed_img):
        mse_loss = (batch['mask'] * (batch['measure_img'] - completed_img[:, 0]) ** 2).sum(-1).sum(-1)

        discr_loss = torch.log(torch.clamp(real_is_real_prob, min=1e-8, max=1-1e-8)) +\
                     torch.log(torch.clamp(1 - gen_is_real_prob, min=1e-8, max=1-1e-8))

        loss = mse_loss + discr_loss

        # loss = None
        # if self.trainer.current_epoch < 5:
        #     loss = mse_loss
        # elif 5 <= self.trainer.current_epoch < 10:
        #     loss = discr_loss
        # else:
        #     loss = mse_loss + discr_loss

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

        opt = {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

        return opt

    def calc_metrics(self):
        metrics = {}

        return metrics

    def log_metrics(self, metrics: dict, type: str):
        on_step = True if type == 'train' else False

        for key in metrics:
            self.log(key + '_' + type, metrics[key], on_step=on_step, on_epoch=True, logger=True)
