import torch
import thop
from common import *
from copy import deepcopy
import yaml
from utils import make_divisible, check_anchor_order, initialize_weights, scale_img, feature_visualization, time_sync, model_info
import logging
import math

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


""" 代码仍然是官方的代码，增加了一些注释去理解，删除一些没有用到的模块，这里仍然使用yolov5l 结构"""


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    """ head 中的 detect层，用来输出最后预测结果 """
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)  # 有多少个检测检测头，一般三个就是20*20， 40*40，80*80 三个尺度
        self.na = len(anchors[0]) // 2  # 目前也是3
        self.grid = [torch.zeros(1)] * self.nl  # 初始化grid cell
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid

        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 用于输出的三个卷积
        self.inplace = inplace

    def forward(self, x):
        """
        x : 每一个检测头的输入，是一个列表[xx,xx,xx]
        """
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20)

            # (bs, 3, 20, 20, 85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # to x(bs,3,20,20,85)

            if not self.training: #inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                # (bs,3,20,20,85)
                y = x[i].sigmoid()

                if self.inplace:
                    # xywh 相对于预测值的转换，这个跟v3 公式不太一样 stride 下采样倍数[8,16,32] 因为实际预测的值是归一化后的结果
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i] #xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953  可以不用管，针对于亚马逊的一个部署的改进
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                # (bs, 3*20*20, 85)
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)  # [(bs, 3*20*20, 85), (bs, 3*40*40, 85), ...] inference 三个检测头结果cat

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])  # 生成二维网络 都是20*20
        # 网格生成
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()  #(1, 3, ny, nx, 2)  对应网格(x,y)

        # (1, self.na=3, ny, nx, 2) 每一个grid cell对应的3个anchor大小
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()

        return grid, anchor_grid


class Model(nn.Module):

    def __init__(self, cfg='yolov5l.yaml', ch=3, nc=None, anchors=None): # model, input channels, number of classes
        super(Model, self).__init__()

        # 加载yaml文件
        with open(cfg, errors='ignore') as f:
            self.yaml = yaml.safe_load(f)  # model dict

        # 获取ch, nc , anchors
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)

        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)

        # 构建模型结构model, 以及存着哪些层的输出会作为其他层的输入[6, 4, 14, 10, 17, 20, 23]
        self.model, self.save = paras_model(deepcopy(self.yaml), [ch])

        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names[0,1,2,...,79]

        # 加速推理的，默认True不使用，
        # AWS Inferentia Inplace compatiability https://github.com/ultralytics/yolov5/pull/2953
        self.inplace = self.yaml.get('inplace', True)

        # build strides, anchors
        m = self.model[-1]  # # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace

            # 获取最终输出feature map 尺度相对于原图尺度下采样倍数[8,16,32] 对应小中大三种尺度的anchor, 目的给anchors做归一化
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)

            # 检查anchor顺序与stride顺序是否一致
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once 初始化权重偏置参数

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        # 使用默认参数 # 默认执行 正常前向推理
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    # 推理时是否采用数据增强上下flip/左右flip
    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs  y存放self.save中每一层的输出，dt在profile打印评估

        # 开始构建模型
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                # 这里的m.f是一个列表，表示来自哪几层作为输入，如[-1,6]则是上一层和第六层作为输入构建新的x, 此时x列表, 只有concat和Detect层是接受列表作为输入的
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

            # 做了一些可
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y


    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p


    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def info(self, verbose=True, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def paras_model(d, ch):
    """
    Args:
        d: yaml dict
        ch: 输入通道数
    Returns: model, save
    """
    LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))

    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)  每一个grid cell预测的数量

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out, ch记录是每一层的输出通道数是多少

    # cong yaml 文件构建模型 构建每一层的输入参数列表args
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args

        m = eval(m) if isinstance(m, str) else m  # 赋予表达式

        for j, a in enumerate(args):

            try:
                args[j] = eval(a) if isinstance(a, str) else a

            except NameError:
                pass

        # 实际层数 要乘上 depth_multiple，然后四舍五入取整
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain

        # 实际上 当前版本只有Conv Bottleneck SPPF C3, 其他在common.py中让其为None
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:

            c1, c2 = ch[f], args[0]   # 当前层，输入输出通道，f有可能是列表

            # 实际输出通道数 要乘上width_multiple / 8 向上取整之后在乘8，有点绕，在l版本中等于自己本身
            if c2 != no:  # if not output  最终输出层不需要做系数相乘
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]  # 构成层的输入参数 c1,c2,k,s,p 对于Conv

            # C3 层参数，需要额外插入重复的bottleneck
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n) # number of repeats
                n = 1  # 对于C3 重复的是bottleneck 而不是整个C3,所以n=1

        # BN层参数只有上一层的通道数，事实上，BN层作为Conv的一部分，没有单独的BN层
        elif m is nn.BatchNorm2d:
            args = [ch[f]]

        elif m is Concat:
            c2 = sum([ch[x] for x in f])  # 在common里也注释过了，Concat实际上是把输入通道数在一起

        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)

        # 以下在当前版本没使用
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2

        # nn.Upsample
        else:
            c2 = ch[f]

        # 按照yaml 从上到下构建每一层模型
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')  # moudel type
        # print(str(m), t)  <class 'common.Conv'> common.Conv
        np = sum([x.numel() for x in m_.parameters()])  # # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params

        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n_, np, t, args))  # print

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        # print(save) # [6, 4, 14, 10, 17, 20, 23] 记录哪些层是作为其他层的输入
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    from torchinfo import summary
    net = Model('yolov5l.yaml')

    summary(net, input_size=(1, 3, 640, 640), depth=4)

