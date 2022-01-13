import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from network.resnet import ResNetBottleneck
from network.mhsa import MHSA, LayerNorm


class FFN(nn.Module):
    """ Feed Forward Network

    """
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class NTL(nn.Module):
    """ Normal Transformer Layer

    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4, sr_ratio=1, qkv_bias=False, drop=0., attn_drop=0., drop_path=0., norm=None, size=None, pos_encoding=False):
        super().__init__()
        norm = norm or nn.Identity
        self.norm1 = norm(dim)
        self.norm2 = norm(dim)

        self.attn = MHSA(dim=dim, num_heads=num_heads, sr_ratio=sr_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop,
                         proj_drop=drop, norm=norm, size=size, pos_encoding=pos_encoding)
        self.ffn = FFN(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x


class NTB(nn.Module):
    """ Normal Transformer Block

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
                NTL(
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


class NTNet(nn.Module):
    """ Normal Transformer Network

    """
    def __init__(self, num_classes, embed_dim, depth, num_heads=8, mlp_ratio=4, sr_ratio=1, qkv_bias=False, drop_path=0.1, pos_encoding=False):
        super().__init__()

        self.patch_embed = ResNetBottleneck(1, embed_dim // 2)
        self.layer = NTB(dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, sr_ratio=sr_ratio,
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
