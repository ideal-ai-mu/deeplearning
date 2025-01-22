import torch.nn as nn
import torch


# 1. 残差块
class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super(ResidualBlock, self).__init__()

        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class VQVAE(nn.Module):
    def __init__(self, input_dim, dim, n_embedding):
        """
        input_dim: 输入通道数，比如3，输入的图片是3通道的
        dim：编码后ze的通道数
        n_embedding：code book 向量的个数
        """
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim),
            ResidualBlock(dim)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim),
            ResidualBlock(dim),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1)
        )
        self.n_downsample = 2

        # code book
        self.vq_embedding = nn.Embedding(n_embedding, dim)
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)

    def forward(self, x):
        """
        x, shape(N,C0,H0,W0)
        """
        # encoder (N,C,H,W)
        ze = self.encoder(x)

        # code book embedding [K, C]
        embedding = self.vq_embedding.weight.data

        N, C, H, W = ze.shape
        K, _ = embedding.shape

        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)

        # 最近距离， 这一步旨在求得zq，这里通过先求ze->z，在求z->zq，事实上z只作为中间变量，通过(zq-ze).detach从计算图分离，避开不能的反向传播
        distance = torch.sum((embedding_broadcast - ze_broadcast) ** 2, 2)  # (N,K,H,W)
        nearest_neghbor = torch.argmin(distance, 1)  # (N,H,W)

        # zq (N, C, H, W) : (N, H, W, C) -> (N, C, H, W)
        zq = self.vq_embedding(nearest_neghbor).permute(0, 3, 1, 2)

        # sg(zq - ze)
        decoder_input = ze + (zq - ze).detach()

        # decoder
        x_hat = self.decoder(decoder_input)

        return x_hat, ze, zq

    # encode z  这一步指在得到离散变量，类似于像素值, 作为输入和标签好用来训练pixel cnn， pixel cnn的目的是用来重建z的，生成z
    @torch.no_grad()
    def encode_z(self, x):
        ze = self.encoder(x)
        embedding = self.vq_embedding.weight.data

        # ze: [N, C, H, W]
        # embedding [K, C]
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast) ** 2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        return nearest_neighbor

    # decode z 这一步指在从pixelcnn得到的结果latent生成最终结果, 因为pixel cnn的结果生成的latent 是离散的z
    @torch.no_grad()
    def decode_z(self, latent_z):
        """
        latent: shape, (N, H, W)
        """
        # zq (N, C, H, W)
        zq = self.vq_embedding(latent_z).permute(0, 3, 1, 2)
        x_hat = self.decoder(zq)
        return x_hat

    # shape: [C,H,W]
    def get_latent_HW(self, input_shape):
        C, H, W = input_shape
        return H // 2 ** self.n_downsample, W // 2 ** self.n_downsample


if __name__ == '__main__':
    from torchinfo import summary

    vqvae = VQVAE(1, 32, 32)
    summary(vqvae, input_size=[1, 1, 28, 28])
