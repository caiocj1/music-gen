import os

import yaml
from yaml.loader import SafeLoader

import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms
import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule

from utils import plot_generated
from models.nn_blocks import CompletionNetwork, LocalDiscriminator, GlobalDiscriminator

class MelodyCompletionNet(LightningModule):

    def __init__(self):
        super(MelodyCompletionNet, self).__init__()
        self.read_config()
        self.build_model()

    def build_model(self):
        self.completion_net = CompletionNetwork()

        self.local_discriminator = LocalDiscriminator()

        self.global_discriminator = GlobalDiscriminator()

        self.discriminate = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def read_config(self):
        config_path = os.path.join(os.getcwd(), 'config.yaml')
        with open(config_path) as f:
            params = yaml.load(f, Loader=SafeLoader)
        training_params = params['TrainingParams']

        self.alpha = training_params['alpha']

    def training_step(self, batch, batch_idx, optimizer_idx):
        loss, training_who, metrics = self._shared_step(batch, optimizer_idx)

        loss = loss.mean()
        self.log_metrics(metrics, 'train')
        self.log(f'{training_who}_loss_train', loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, training_who, metrics = self._shared_step(batch, 0)

        loss = loss.mean()
        self.log_metrics(metrics, 'val')
        self.log(f'{training_who}_loss_val', loss, on_step=False, on_epoch=True, logger=True)

        batch_size = len(self.completed_img)
        for i in range(batch_size):
            idx = batch_size * batch_idx + i
            if idx % 10 == 0:
                fig = plot_generated(batch, self.completed_img, None, None, i)
                img_tensor = torchvision.transforms.ToTensor()(fig)
                self.logger.experiment.add_image(f'generated_imgs/img_{idx}', img_tensor, self.global_step)

        return loss

    def _shared_step(self, batch, optimizer_idx):
        loss = None
        training_who = 'g'

        # train generator
        if optimizer_idx == 0:
            self.completed_img = self.forward(batch)
            loss = self.calc_g_loss(batch, self.completed_img)

        # train discriminator
        if optimizer_idx == 1:
            gen_is_real_prob, real_is_real_prob = self.discr_forward(batch, self.completed_img)
            loss = self.calc_d_loss(gen_is_real_prob.detach(), real_is_real_prob)
            training_who = 'd'

        metrics = self.calc_metrics()

        return loss, training_who, metrics

    def forward(self, batch):
        masked_input = torch.stack((batch['measure_img'] * (1 - batch['mask']), batch['mask']), dim=1).float()
        completed_img = self.completion_net(masked_input)

        return completed_img

    def discr_forward(self, batch, completed_img):
        batch_size = len(completed_img)

        gen_local_img = completed_img[batch['mask'][:, None].bool()].reshape(batch_size, 1, 128, 100)
        gen_local_vec = self.local_discriminator(gen_local_img)
        # gen_global_vec = self.global_discriminator(completed_img)
        gen_global_vec = self.global_discriminator((completed_img * batch['mask'][:, None] +
                                                    batch['measure_img'][:, None] * (
                                                                1 - batch['mask'][:, None])).float())
        gen_discr_vec = torch.cat((gen_local_vec, gen_global_vec), dim=1)
        gen_is_real_prob = self.discriminate(gen_discr_vec)[:, 0]

        real_local_img = batch['measure_img'][batch['mask'].bool()].reshape(batch_size, 1, 128, 100).float()
        real_local_vec = self.local_discriminator(real_local_img)
        real_global_vec = self.global_discriminator(batch['measure_img'][:, None].float())
        real_discr_vec = torch.cat((real_local_vec, real_global_vec), dim=1)
        real_is_real_prob = self.discriminate(real_discr_vec)[:, 0]

        return gen_is_real_prob, real_is_real_prob

    def calc_g_loss(self, batch, completed_img):
        g_loss = (batch['mask'] * (batch['measure_img'] - completed_img[:, 0]) ** 2).sum(-1).sum(-1)
        return g_loss

    def calc_d_loss(self, gen_is_real_prob, real_is_real_prob):
        d_loss = (torch.log(torch.clamp(real_is_real_prob, min=1e-8, max=1 - 1e-8)) + \
                      self.alpha * torch.log(torch.clamp(1 - gen_is_real_prob, min=1e-8, max=1 - 1e-8)))
        return d_loss

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adadelta(self.parameters(), lr=0.1)
        d_optimizer = torch.optim.Adadelta(self.parameters(), lr=0.1, maximize=True)
        #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=0.98)

        # opt = {
        #     'optimizer': [g_optimizer, d_optimizer],
        #     'state': []
        #     #'lr_scheduler': lr_scheduler
        # }

        return g_optimizer, d_optimizer

    def calc_metrics(self):
        metrics = {}

        return metrics

    def log_metrics(self, metrics: dict, type: str):
        on_step = True if type == 'train' else False

        for key in metrics:
            self.log(key + '_' + type, metrics[key], on_step=on_step, on_epoch=True, logger=True)
