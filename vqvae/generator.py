import torch
import einops
import cv2
import numpy as np
import torchvision
from pixelcnnmodel import GatedPixelCNN, PixelCnnWithEmbedding
from vqvaemodel import VQVAE
from torchvision import transforms
from torch.utils.data import DataLoader

# 看一下vae 的效果
def reconstruct(model, x, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x_hat, _, _ = model(x)

    n = x.shape[0]
    n1 = int(n ** 0.5)
    x_cat = torch.concat((x, x_hat), 3)
    x_cat = einops.rearrange(x_cat, '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=n1)
    x_cat = (x_cat.clip(0, 1) * 255).cpu().numpy().astype(np.uint8)
    cv2.imwrite(f'ckpt/reconstruct_show.jpg', x_cat)


# 看一下最终生成的效果
def sample_imgs(
        vqvae: VQVAE,
        gen_model,
        img_shape,
        device,
        n_sample=81
):
    vqvae = vqvae.to(device)
    gen_model = gen_model.to(device)

    vqvae.eval()
    gen_model.eval()

    # 获取latent space H,W
    C, H, W = img_shape
    H, W = vqvae.get_latent_HW((C, H, W))

    input_shape = (n_sample, H, W)
    latent_z = torch.zeros(input_shape).to(device).to(torch.long)
    # pixel cnn sample
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                output = gen_model(latent_z)
                prob_dist = torch.softmax(output[:, :, i, j], -1)
                pixel = torch.multinomial(prob_dist, 1)
                latent_z[:, i, j] = pixel[:, 0]

    # vqvae decode 由z->x_hat
    imgs = vqvae.decode_z(latent_z)

    imgs = imgs * 255
    imgs = imgs.clip(0, 255)
    imgs = einops.rearrange(imgs,
                            '(n1 n2) c h w -> (n1 h) (n2 w) c',
                            n1=int(n_sample ** 0.5))
    imgs = imgs.detach().cpu().numpy().astype(np.uint8)
    cv2.imwrite('ckpt/sample_show.jpg', imgs)


def test():
    # 训练完成，测试一下效果
    device = torch.device("cuda:0")

    mnist = torchvision.datasets.MNIST(root='./data/mnist', download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(mnist, batch_size=64, shuffle=True)
    batch_imgs, _ = next(iter(dataloader))

    # vqvae
    vqvae = VQVAE(1, 32, 32)
    vqvae.load_state_dict(torch.load('ckpt/vqvae_ckpt.pth'))
    vqvae.eval()
    vqvae = vqvae.to(device)
    batch_imgs = batch_imgs.to(device)
    reconstruct(vqvae, batch_imgs, device)

    gen_model = PixelCnnWithEmbedding(15, 128, 32)
    gen_model.load_state_dict(torch.load('ckpt/gen_ckpt.pth'))
    gen_model.eval()
    gen_model = gen_model.to(device)
    sample_imgs(vqvae, gen_model, (1, 28, 28), device)


if __name__ == '__main__':
    test()
