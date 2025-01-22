import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from vqvaemodel import VQVAE
from pixelcnnmodel import GatedPixelCNN, PixelCnnWithEmbedding


# 依然拿mnist 作为数据集
# 看一下mnist的样子
def mnist_show():
    mnist = torchvision.datasets.MNIST(root='./data/mnist', download=True)
    print('length of MNIST', len(mnist))
    img, label = mnist[0]
    print(img)
    print(label)
    img.show()
    tensor = transforms.ToTensor()(img)
    print(tensor.shape)  # torch.Size([1, 28, 28])  CHW
    print(tensor.max())  # max 1,
    print(tensor.min())  # min 0, 已经是归一化的结果



def train_vqvae(
        model: VQVAE,
        device,
        dataloader,
        ckpt_vqvae='ckpt/vqvae_ckpt.pth',
        n_epochs=100,
        alpha=1,
        beta=0.25,
):
    model.to(device)  # model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = torch.nn.MSELoss()

    print("start vqvae train...")
    for epo in range(n_epochs):

        for img, label in dataloader:
            x = img.to(device)  # N1HW
            x_hat, ze, zq = model(x)

            # ||x - decoder(ze+sg(zq-ze))||
            loss_rec = mse_loss(x, x_hat)

            # ||zq - sg(ze)||
            loss_zq = mse_loss(zq, ze.detach())

            # ||sg(zq) - ze||
            loss_ze = mse_loss(zq.detach(), ze)

            loss = loss_rec + alpha * loss_zq + beta * loss_ze

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"epoch:{epo}, loss:{loss.item():.6f}")

        if epo % 10 == 0:
            torch.save(model.state_dict(), ckpt_vqvae)

    print("vqvae train finish!!")


def train_gen(
        vqvae: VQVAE,
        model,
        device,
        dataloader,
        ckpt_gen="ckpt/gen_ckpt.pth",
        n_epochs=50,
):
    vqvae = vqvae.to(device)
    model = model.to(device)
    vqvae.eval()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("start pixel cnn train...")
    for epo in range(n_epochs):

        for x, _ in dataloader:
            with torch.no_grad():
                x = x.to(device)

                # 得到离散变量z
                z = vqvae.encode_z(x)

            # 使用pixel cnn重建这个离散变量z,记住是重建的z 而非x 即由z->z
            predict_z = model(z)
            loss = loss_fn(predict_z, z)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"epoch:{epo}, loss:{loss.item():.6f}")

        if epo % 10 == 0:
            torch.save(model.state_dict(), ckpt_gen)

    print("pixel train finish!!")


def main():
    """ 代码中的公式符号尽可能和原论文一致，避免混淆，尤其是ze,z,zq这几个概念 """
    device = torch.device("cuda:0")
    mnist = torchvision.datasets.MNIST(root='./data/mnist', download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(mnist, batch_size=512, shuffle=True)

    # 0. 构建模型
    vqvae = VQVAE(1, 32, 32)
    gen_model = PixelCnnWithEmbedding(15, 128, 32)

    # 1. train vqvae , reconstruct
    train_vqvae(vqvae, device, dataloader)

    # 2. train gen model, sample
    vqvae.load_state_dict(torch.load('ckpt/vqvae_ckpt.pth'))
    train_gen(vqvae, gen_model, device, dataloader)



if __name__ == '__main__':
    main()

