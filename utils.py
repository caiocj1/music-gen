import matplotlib.pyplot as plt
from PIL import Image
import io

def plot_generated(batch, completed_img, gen_is_real_prob, real_is_real_prob, idx, plt_figure = False):
    fig, axs = plt.subplots(4)
    fig.set_size_inches(5.5, 8)

    fontdict = {'fontsize': 10}

    axs[0].set_title('Real image', fontdict=fontdict)
    axs[0].imshow(batch['measure_img'][idx].cpu().detach().numpy())
    axs[0].axis('off')
    axs[1].set_title('Masked image: generator input', fontdict=fontdict)
    axs[1].imshow((batch['measure_img'][idx] * (1 - batch['mask'][idx])).cpu().detach().numpy())
    axs[1].axis('off')
    axs[2].set_title('Completed image: whole', fontdict=fontdict)
    axs[2].imshow(completed_img[idx, 0].cpu().detach().numpy())
    axs[2].axis('off')
    axs[3].set_title('Completed image: discriminator input', fontdict=fontdict)
    axs[3].imshow((completed_img[idx, 0] * batch['mask'][idx] + batch['measure_img'][idx] * (
                1 - batch['mask'][idx])).cpu().detach().numpy())
    if gen_is_real_prob is not None and real_is_real_prob is not None:
        axs[3].annotate('[{:.4f} {:.4f}]'.format(gen_is_real_prob.item(), real_is_real_prob.item()), (10, 14))
    axs[3].axis('off')

    # pil_img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    # plt.close(fig)
    if plt_figure:
        return fig

    buf = io.BytesIO()
    fig.savefig(buf, dpi=500)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img