import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from simplegan import Generator, Discriminator
from tqdm import tqdm
import cv2
import numpy as np


# 超参数
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1
batch = 32

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ]
)


def train(num_epochs):
    Disc = Discriminator(image_dim).to(device)
    Gen = Generator(z_dim, image_dim).to(device)
    opt_disc = optim.Adam(Disc.parameters(), lr=lr)
    opt_gen = optim.Adam(Gen.parameters(), lr=lr)
    criterion = nn.BCELoss()  # 单目标二分类交叉熵函数

    dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
    loader = DataLoader(dataset=dataset, batch_size=batch, shuffle=True)

    fixed_noise = torch.randn((batch, z_dim)).to(device)

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in tqdm(enumerate(loader)):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]
            ## D: 目标：真的判断为真，假的判断为假
            ## 训练Discriminator: max log(D(x)) + log(1-D(G(z)))
            disc_real = Disc(real)#.view(-1)  # 将真实图片放入到判别器中
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))  # 真的判断为真

            noise = torch.randn(batch_size, z_dim).to(device)
            fake = Gen(noise)  # 将随机噪声放入到生成器中
            disc_fake = Disc(fake).view(-1)  # 判别器判断真假
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))  # 假的应该判断为假
            lossD = (lossD_real + lossD_fake) / 2  # loss包括判真损失和判假损失

            Disc.zero_grad()   # 在反向传播前，先将梯度归0
            lossD.backward(retain_graph=True)  # 将误差反向传播
            opt_disc.step()   # 更新参数

            # G： 目标：生成的越真越好
            ## 训练生成器： min log(1-D(G(z))) <-> max log(D(G(z)))
            output = Disc(fake).view(-1)   # 生成的放入识别器
            lossG = criterion(output, torch.ones_like(output))  # 与“真的”的距离，越小越好
            Gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            # 输出一些信息，便于观察
            if batch_idx % 20 == 0:

                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)}' \
                        loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

                # 推理生成结果
                with torch.no_grad():
                    fake = Gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                    img_grid_combined = torch.cat((img_grid_real, img_grid_fake), dim=2)
                    img_grid_combined = img_grid_combined.permute(1, 2, 0).cpu().detach().numpy()
                    img_grid_combined = (img_grid_combined * 255).astype(np.uint8)
                    # 使用 cv2 显示图片
                    cv2.imshow('Combined Image', img_grid_combined)
                    cv2.waitKey(1)
                    cv2.imwrite(f"ckpt/tmp_simple.jpg", img_grid_combined)

        # 10个epoch进行一次模型保存, 实际上只用保存生成器Gen模型即可
        if (epoch + 1) % 10 == 0:
            torch.save(Gen.state_dict(), f"ckpt/simple_gen_weights.pth")
            torch.save(Disc.state_dict(), f"ckpt/simple_disc_weights.pth")


if __name__ == '__main__':
    train(100)
