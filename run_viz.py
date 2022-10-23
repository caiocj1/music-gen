import argparse

from models.melody_completion_net import MelodyCompletionNet
from dataset import MusicDataset

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='Maestro')
    parser.add_argument('--weights_path', '-w')

    args = parser.parse_args()

    eval_dataset = MusicDataset(dataset=args.dataset, type='validation', max_samples=2)
    eval_dataloader = DataLoader(eval_dataset,
                                batch_size=1,
                                num_workers=0,
                                shuffle=False)

    model = MelodyCompletionNet()
    if args.weights_path is not None:
        ckpt = torch.load(args.weights_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        print('loaded weights')
    model.eval()

    for _, batch in enumerate(eval_dataloader):
        print(_, '/', len(eval_dataloader))
        gen_is_real_prob, real_is_real_prob, completed_img = model(batch)

        fig, axs = plt.subplots(4)
        fig.set_size_inches(8, 16)
        fontdict = {'fontsize': 10}
        axs[0].set_title('Real image', fontdict=fontdict)
        axs[0].imshow(batch['measure_img'][0].cpu().detach().numpy())
        axs[0].axis('off')
        axs[1].set_title('Masked image: generator input', fontdict=fontdict)
        axs[1].imshow((batch['measure_img'][0] * (1 - batch['mask'][0])).cpu().detach().numpy())
        axs[1].axis('off')
        axs[2].set_title('Completed image: whole', fontdict=fontdict)
        axs[2].imshow(completed_img[0, 0].cpu().detach().numpy())
        axs[2].axis('off')
        axs[3].set_title('Completed image: discriminator input', fontdict=fontdict)
        axs[3].imshow((completed_img[0, 0] * batch['mask'][0] + batch['measure_img'][0] * (1 - batch['mask'][0])).cpu().detach().numpy())
        axs[3].annotate('[{:.4f} {:.4f}]'.format(gen_is_real_prob.item(), real_is_real_prob.item()), (10, 14))
        axs[3].axis('off')
        plt.show()
