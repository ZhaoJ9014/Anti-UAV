import numpy as np
#from efficientnet_pytorch import EfficientNet
from jittor import nn
from ..backbone import *
import jittor as jt

class Extra_fq(jt.nn.Module):
    """
    img_size : the size of train image
    out_c : the channel of feature map extracted by this block(the channel mast be same as the spatial domain feature map)
    """
    def __init__(self, img_size, out_c):
        super(Extra_fq, self).__init__()
        self.img_size = img_size
        #self.batch_size = batch_size

        def get_DCTf(img_size):
            timg = jt.zeros((img_size, img_size))

            M, N = jt.var(np.array(img_size)), jt.var(np.array(img_size))
            timg[0, :] = 1 * jt.sqrt(1 / N)
            for i in range(1, M):
                for j in range(N):
                    timg[i, j] = jt.cos(np.pi * i * (2 * j + 1) / (2 * N)) * jt.sqrt(2 / N)
            return timg



        def create_f(img_size):
            resolution = (img_size, img_size)
            low_f = jt.zeros(resolution)
            high_f = jt.ones(resolution)
            mid_f = jt.ones(resolution)
            resolution = np.array(resolution)
            t_1 = resolution // 16
            t_2 = resolution // 8
            for i in range(t_1[0]):
                for j in range(t_1[1] - i):
                    low_f[i, j] = 1
            for i in range(t_2[0]):
                for j in range(t_2[1]-i):
                    high_f[i, j] = 0
            mid_f = mid_f - low_f
            mid_f = mid_f - high_f
            return low_f, mid_f, high_f

        self.dct_filter = get_DCTf(img_size=img_size)

        self.low_f, self.mid_f, self.high_f = create_f(img_size=img_size)


        self.block = nn.Sequential(
            jt.nn.Conv2d(3, int(out_c / 2), kernel_size=3, stride=1, padding=1),
            jt.nn.BatchNorm2d(int(out_c / 2)),
            jt.nn.ReLU(inplace=True),
            jt.nn.MaxPool2d(kernel_size=2, stride=2),

            jt.nn.Conv2d(int(out_c / 2), out_c, kernel_size=3, stride=1, padding=1),
            jt.nn.BatchNorm2d(out_c),
            jt.nn.ReLU(inplace=True),
            jt.nn.MaxPool2d(kernel_size=2, stride=2)



        )

    def DCT(self, img):

        dst = self.dct_filter * img
        dst = dst * self.dct_filter.permute(0, 1)
        return dst

    def IDCT(self, img):

        dst = self.dct_filter.permute(0, 1) * img
        dst = dst * self.dct_filter
        return dst

    def forward(self, img):
        r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
        dct_1, dct_2, dct_3 = self.DCT(r), self.DCT(g), self.DCT(b)

        fl = [self.low_f, self.mid_f, self.high_f]
        re = []
        for i in range(3):
            t_1 = dct_1 * fl[i]
            t_2 = dct_2 * fl[i]
            t_3 = dct_3 * fl[i]
            re.append(self.IDCT(t_1 + t_2 + t_3))
        out = jt.concat((re[0].unsqueeze(1), re[1].unsqueeze(1), re[2].unsqueeze(1)), dim=1)

        out = self.block(out)
        return out


class PatchEmbed(jt.nn.Module):
    """
    img_size : the size of feature map extracted by spatial domain
    patch_size : the size of the patch using on embeding
    in_c : the channel of the feature map extracted by spatial domain
    embed_dim:the embeding dimension
    """
    def __init__(self, img_size=56, patch_size=56, in_c=32, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = jt.nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else jt.nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x  # [b, p_num, embed_dim]


class ATT(jt.nn.Module):
    """
    in_c : the channel of the feature map extracted by spatial domain
    patch_size : the size of the patch using on embeding
    dim : the embeding dimension
    attn_drop_ratio : the dropout rate of the attention
    proj_drop_ratio : the dropout rate of the projection
    """
    def __init__(self,

                 in_c,
                 patch_size,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(ATT, self).__init__()
        self.num_heads = num_heads
        self.patch_size = patch_size
        # self.in_c = in_c
        #head_dim = dim // num_heads
        self.scale = (patch_size * patch_size * in_c) ** -0.5
        self.qkv = jt.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = jt.nn.Dropout(attn_drop_ratio)
        self.proj = jt.nn.Linear(dim, patch_size * patch_size)
        self.proj_drop = jt.nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.reshape((B, N, self.patch_size, self.patch_size))
        return x


class MSMHTR(jt.nn.Module):
    """
    img_size : the size of feature map extracted by spatial domain
    in_c : the channel of feature map extracted by spatial domain
    n_dim : the embed dimension of the patch embed
    drop_ratio : the dropout rate
    """
    def __init__(self,
                 img_size,
                 in_c,
                 n_dim,
                 drop_ratio):
        super(MSMHTR, self).__init__()
        self.patch_size = img_size
        self.in_c = in_c

        num_patches = int((img_size // self.patch_size) ** 2)
        # self.cls1 = nn.Parameter(torch.zeros(1, 1, n_dim))
        self.pos1 = jt.nn.Parameter(jt.zeros(1, num_patches, n_dim))
        self.eb1 = PatchEmbed(patch_size=self.patch_size, embed_dim=n_dim, in_c=self.in_c)
        self.att1 = ATT(in_c=in_c, dim=n_dim, patch_size=self.patch_size)
        self.patch_size = int(self.patch_size / 2)

        num_patches = int((img_size // self.patch_size) ** 2)
        # self.cls2 = nn.Parameter(torch.zeros(1, 1, n_dim))
        self.pos2 = jt.nn.Parameter(jt.zeros(1, num_patches, n_dim))
        self.eb2 = PatchEmbed(patch_size=self.patch_size, embed_dim=n_dim, in_c=self.in_c)
        self.att2 = ATT(in_c=in_c, dim=n_dim, patch_size=self.patch_size)
        self.patch_size = int(self.patch_size / 2)

        num_patches = int((img_size // self.patch_size) ** 2)
        # self.cls3 = nn.Parameter(torch.zeros(1, 1, n_dim))
        self.pos3 = jt.nn.Parameter(jt.zeros(1, num_patches, n_dim))
        self.eb3 = PatchEmbed(patch_size=self.patch_size, embed_dim=n_dim, in_c=self.in_c)
        self.att3 = ATT(in_c=in_c, dim=n_dim, patch_size=self.patch_size)
        self.patch_size = int(self.patch_size / 2)

        num_patches = int((img_size // self.patch_size) ** 2)
        # self.cls4 = nn.Parameter(torch.zeros(1, 1, n_dim))
        self.pos4 = jt.nn.Parameter(jt.zeros(1, num_patches, n_dim))
        self.eb4 = PatchEmbed(patch_size=self.patch_size, embed_dim=n_dim, in_c=self.in_c)
        self.att4 = ATT(in_c=in_c, dim=n_dim, patch_size=self.patch_size)

        self.pos_drop = jt.nn.Dropout(p=drop_ratio)

    def forward(self, x):
        eb1 = self.eb1(x)
        input1 = self.pos_drop(eb1 + self.pos1)
        att1 = self.att1(input1)

        eb2 = self.eb2(x)
        input2 = self.pos_drop(eb2 + self.pos2)
        att2 = self.att2(input2)
        att2 = att2.reshape(att1.shape)

        eb3 = self.eb3(x)
        input3 = self.pos_drop(eb3 + self.pos3)
        att3 = self.att3(input3)
        att3 = att3.reshape(att1.shape)

        eb4 = self.eb4(x)
        input4 = self.pos_drop(eb4 + self.pos4)
        att4 = self.att4(input4)
        att4 = att4.reshape(att1.shape)

        return att1 + att2 + att3 + att4


class CMF(jt.nn.Module):
    """
    in_c : the channel of the feature map stacked by spatial domain, frequency domain and MSMHTR
    img_size : the size of the feature map stacked by spatial domain, frequency domain and MSMHTR
    """

    def __init__(self,
                 in_c,
                 img_size=56):
        super(CMF, self).__init__()
        self.convq = jt.nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)
        self.convk = jt.nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)
        self.convv = jt.nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)
        self.scale = (img_size * img_size * in_c) ** -0.5

        self.conv1 = jt.nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=1)

    def forward(self, x_s, x_fq, x_mt):
        q = self.convq(x_s)
        k = self.convk(x_fq)
        v = self.convv(x_fq)
        fuse = (q @ k.transpose(-2, -1)) * self.scale
        fuse = fuse.softmax(dim=-1)
        fuse = fuse @ v

        f_cmf = self.conv1(x_s + x_mt + fuse)

        return f_cmf


class M2TR(nn.Module):
    """
    img_size : the size of your train img
    n_dim :  the dimension of patch embeding
    drop_ratio :  the dropout rate
    """
    def __init__(self, img_size, n_dim, drop_ratio):
        super(M2TR, self).__init__()
      #  self.model = EfficientNet.from_name('efficientnet-b4')
        self.model = resnet50()
       # state_dict = torch.load(r'C:\Users\satomi ishihara\za\desktop\fakeface\efficientnet-b4.pth')
        #self.model.load_state_dict(state_dict)
        self.backbone1 = jt.nn.Sequential(
            jt.nn.PixelShuffle(2),
            jt.nn.PixelShuffle(2),
            jt.nn.PixelShuffle(2)
        )

        self.ex_fq = Extra_fq(img_size=img_size, out_c=28)
        self.mt = MSMHTR(img_size=int(img_size / 4), in_c=28, n_dim=n_dim, drop_ratio=drop_ratio)

        self.cmf = CMF(in_c=28, img_size=img_size / 4)

        self.backbone2 = jt.nn.Sequential(
            jt.nn.Conv2d(28, 32, kernel_size=3, stride=1, padding=1),
            jt.nn.BatchNorm2d(32),
            jt.nn.ReLU(inplace=True),
            jt.nn.MaxPool2d(kernel_size=2, stride=2),

            jt.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            jt.nn.BatchNorm2d(64),
            jt.nn.ReLU(inplace=True),
            jt.nn.MaxPool2d(kernel_size=2, stride=2),

            jt.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            jt.nn.BatchNorm2d(128),
            jt.nn.ReLU(inplace=True),
            jt.nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.fc = jt.nn.Sequential(
            jt.nn.Linear(int(((img_size / 32) ** 2) * 128), 512),
            jt.nn.Linear(512, 2),
            jt.nn.ReLU(inplace=True)
        )

    def feature_forward(self, x):
        return self.backbone1(self.model.extract_features(x))

    def forward(self, x):
        #x_ = self.model.extract_features(x)
        f_s = self.feature_forward(x)
        f_fq = self.ex_fq(x)
        f_mt = self.mt(f_s)
        f_cmf = self.cmf(f_s, f_fq, f_mt)

        out = self.backbone2(f_cmf)
        out = jt.flatten(out, 1)
        out = self.fc(out)
        return out.softmax(dim=-1)  # , f_s
def binary_cross_entropy(input, target):
    # 计算二进制交叉熵损失
    loss = -target * jt.log(input) - (1 - target) * jt.log(1 - input)
    return jt.mean(loss)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = jt.nn.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = binary_cross_entropy(inputs, targets, reduce=False)
        pt = jt.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return jt.mean(F_loss)
        else:
            return F_loss

def create_FocalLoss(alpha, gamma, logits=False, reduce=True):
    return FocalLoss(alpha=alpha, gamma=gamma, logits=logits, reduce=reduce)


def create_model(img_size=224, n_dim=768, drop_ratio=0.1):
    #modify hera to change your img_size, embed_dim, and drop_ratio
    return  M2TR(img_size=img_size, n_dim=n_dim, drop_ratio=drop_ratio)