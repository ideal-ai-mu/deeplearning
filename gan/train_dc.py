import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from dcgan import Generator, Discriminator, initialize_weights
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
NUM_EPOCHS = 5
CHANNELS_IMG = 1
NOISE_DIM = 100
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),

        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)


def train(NUM_EPOCHS):

    # 数据load
    dataset = MNIST(root='./data', train=True, download=True, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)

    gen.train()
    disc.train()

    for epoch in range(NUM_EPOCHS):
        # 不需要目标的标签，无监督
        for batch_id, (real, _) in enumerate(dataloader):
            real = real.to(device)
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
            fake = gen(noise)

            # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            disc_real = disc(real).reshape(-1)
            loss_real = criterion(disc_real, torch.ones_like(disc_real))

            # 训练判别器，生成器输出的值从计算图剥离出来
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            loss_disc = (loss_real + loss_fake) / 2

            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)), 先训练一个epoch 的D
            if epoch >= 0:
                output = disc(fake).reshape(-1)
                loss_gen = criterion(output, torch.ones_like(output))

                gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()

                if batch_id % 20 == 0:
                    print(
                        f'Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_id}/{len(dataloader)} Loss D: {loss_disc}, loss G: {loss_gen}')

                    # 推理生成结果
                    with torch.no_grad():
                        fake = gen(fixed_noise)
                        img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                        img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                        img_grid_combined = torch.cat((img_grid_real, img_grid_fake), dim=2)
                        img_grid_combined = img_grid_combined.permute(1, 2, 0).cpu().detach().numpy()
                        img_grid_combined = (img_grid_combined * 255).astype(np.uint8)
                        # 使用 cv2 显示图片
                        cv2.imshow('Combined Image', img_grid_combined)
                        cv2.waitKey(1)
                        cv2.imwrite(f"ckpt/tmp_dc.jpg", img_grid_combined)

        if (epoch + 1) % 10 == 0:
            torch.save(gen.state_dict(), f"ckpt/dc_gen_weights.pth")
            torch.save(disc.state_dict(), f"ckpt/dc_disc_weights.pth")


if __name__ == "__main__":
    train(100)
