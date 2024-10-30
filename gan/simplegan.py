import torch
import torch.nn as nn
from torchinfo import summary


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 256),  # 784 -> 256
            nn.LeakyReLU(0.2),  #
            nn.Linear(256, 256), # 256 -> 256
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),  # 255 -> 1
            nn.Sigmoid(),   # 将实数映射到[0,1]区间
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, image_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),   # 64 升至 256维
            nn.ReLU(True),
            nn.Linear(256, 256),   # 256 -> 256
            nn.ReLU(True),
            nn.Linear(256, image_dim),  # 256 -> 784
            nn.Tanh(),  # Tanh使得生成数据范围在[-1, 1]，因为真实数据经过transforms后也是在这个区间
        )

    def forward(self, x):
        return self.gen(x)


if __name__ == "__main__":
    gnet = Generator(64, 784)
    dnet = Discriminator(784)

    summary(gnet, input_data=[torch.randn(10, 64)])
    summary(dnet, input_data=[torch.randn(10, 784)])
