import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from network.resnet import ResNetBottleneck
from network.mhsa import LayerNorm, MHSA


class LPU(nn.Module):
    """ Local Perception Uint

    """
    def __init__(self, dim, act=False):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm2d(dim)
        ) if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.dwconv(x))
        return x


class IRFFN(nn.Module):
    """ Inverted Residual Feed Forward Network

    """
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class CMTL(nn.Module):
    """ Convolution Meet Transformer Layer

    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4, sr_ratio=1, qkv_bias=False, drop=0., attn_drop=0., drop_path=0., norm=None, size=None, pos_encoding=False):
        super().__init__()
        norm = norm or nn.Identity
        self.norm1 = norm(dim)
        self.norm2 = norm(dim)

        self.lpu = LPU(dim=dim, act=True)
        self.attn = MHSA(dim=dim, num_heads=num_heads, sr_ratio=sr_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop,
                         proj_drop=drop, norm=norm, size=size, pos_encoding=pos_encoding)
        self.ffn = IRFFN(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.lpu(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x


class CMTB(nn.Module):
    """ Convolution Meet Transformer Block

    """
    def __init__(self, dim, depth, num_heads=8, mlp_ratio=4, sr_ratio=1, qkv_bias=False, drop=0., attn_drop=0., drop_path=0., norm=None, size=None, pos_encoding=False):
        super().__init__()
        norm = norm or nn.Identity
        self.norm1 = norm(dim)
        self.norm2 = norm(dim)

        layers = []
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        for i in range(depth):
            layers.append(
                CMTL(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    sr_ratio=sr_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm=norm,
                    size=size,
                    pos_encoding=pos_encoding
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.norm2(self.layer(self.norm1(x)))

        return x


class CMTNet(nn.Module):
    """ Convolution Meet Transformer Network

    """
    def __init__(self, num_classes, embed_dim, depth, num_heads=8, mlp_ratio=4, sr_ratio=1, qkv_bias=False, drop_path=0.1, pos_encoding=False):
        super().__init__()

        self.patch_embed = ResNetBottleneck(1, embed_dim // 2)
        self.layer = CMTB(dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, sr_ratio=sr_ratio,
                         qkv_bias=qkv_bias, drop_path=drop_path, norm=LayerNorm, size=(56, 56), pos_encoding=pos_encoding)

        self.head = nn.Linear(embed_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.patch_embed(x)
        x = self.layer(x)
        x = x.mean(-1).mean(-1)
        x = self.head(x)
        x = self.softmax(x)

        return x
