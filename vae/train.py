import cv2
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchinfo import summary
import numpy as np

""" linear vae """

CHANNELS_IMG = 1

transforms = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)


class VAE(nn.Module):
    def __init__(self, image_size=28*28, hidden1=512, hidden2=128, latent_dims=20):
        super().__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(image_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(hidden2, latent_dims),
        )

        self.logvar = nn.Sequential(
            nn.Linear(hidden2, latent_dims),
        )   # 由于方差是非负的，因此预测方差对数

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, image_size),
            nn.Sigmoid()
        )

    # 重参数，为了可以反向传播
    def reparametrization(self, mu, logvar):
        # sigma = exp(0.5 * log(sigma^2))= exp(0.5 * log(var))
        std = torch.exp(0.5* logvar)
        # N(mu, std^2) = N(0, 1) * std + mu
        z = torch.randn(std.size(), device=mu.device) * std + mu
        return z

    def forward(self, x):
        en = self.encoder(x)
        mu = self.mu(en)
        logvar = self.logvar(en)
        z = self.reparametrization(mu, logvar)

        return self.decoder(z), mu, logvar


def loss_function(fake_imgs, real_imgs, mu, logvar, criterion, batch):

    kl = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu ** 2) / batch
    reconstruction = criterion(fake_imgs, real_imgs) / batch

    return kl, reconstruction


def train(num_epoch, batch):


    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True)

    vae = VAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.0003)


    # 构建损失函数, 使用BCE来构建重建损失
    criterion = nn.BCELoss(reduction='sum')

    for epoch in range(num_epoch):
        vae.train()
        for batch_indx, (inputs, _) in enumerate(trainloader):

            current_batch = inputs.shape[0]
            inputs = inputs.to(device)

            real_imgs = torch.flatten(inputs, start_dim=1)

            fake_imgs, mu, logvar = vae(real_imgs)

            loss_kl, loss_re = loss_function(fake_imgs, real_imgs, mu, logvar, criterion, current_batch)

            loss_all = loss_kl + loss_re

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            # 每20次迭代打印一次 loss 结果
            if batch_indx % 20 == 0:
                print(f"epoch:{epoch}, loss kl:{loss_kl.item()}, loss re:{loss_re.item()}, loss all:{loss_all.item()}")

        # 每epoch打印推理生成的结果
        vae.eval()
        with torch.no_grad():
            x = torch.randn((32, 20)).to(device)
            fake = vae.decoder(x).reshape(-1, 1, 28, 28)
            data = inputs[:32]
            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            img_grid_real = torchvision.utils.make_grid(data, normalize=True)

            img_grid_combined = torch.cat((img_grid_real, img_grid_fake), dim=2)
            img_grid_combined = img_grid_combined.permute(1, 2, 0).cpu().detach().numpy()
            img_grid_combined = (img_grid_combined * 255).astype(np.uint8)

            # 使用 cv2 显示图片
            cv2.imshow('Combined Image', img_grid_combined)
            cv2.waitKey(1)
            cv2.imwrite(f"ckpt/tmp_vae_t.jpg", img_grid_combined)

        # 每10个epoch 保存一下模型结果
        if (epoch + 1) % 10 == 0:
            torch.save(vae.state_dict(), f"ckpt/vae_weights.pth")



if __name__ == "__main__":
    summary(VAE(), input_size=(1, 784))
    train(num_epoch=1000,
          batch=256)