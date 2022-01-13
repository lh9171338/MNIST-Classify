import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from network.resnet import ResNetBottleneck
from network.mhsa import LayerNorm
from network.ntn import FFN


class EA(nn.Module):
    """ Exemplar Attention

    """
    def __init__(self, dim, query_dim, key_dim, qkv_bias=False):
        super().__init__()
        self.scale = key_dim ** -0.5

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.query = nn.Linear(dim, query_dim, bias=qkv_bias)
        self.key = nn.Linear(query_dim, key_dim, bias=qkv_bias)
        self.value = nn.Conv2d(dim, key_dim * dim, 3, padding=1, bias=qkv_bias)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.avg_pool(x).view(B, -1)
        q = self.query(q)
        attn = self.key(q)
        v = self.value(x).view(B, -1, C, H, W)

        attn = self.softmax(attn * self.scale)
        out = torch.einsum('be, bechw -> bchw', attn, v)

        return out


class ETL(nn.Module):
    """ Exemplar Transformer Layer

    """
    def __init__(self, dim, query_dim, key_dim, mlp_ratio=4., qkv_bias=False, drop_path=0., norm=None):
        super().__init__()
        norm = norm or nn.Identity
        self.norm1 = norm(dim)
        self.norm2 = norm(dim)

        self.attn = EA(dim, query_dim, key_dim, qkv_bias)
        self.ffn = FFN(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x


class ETB(nn.Module):
    """ Exemplar Transformer Block

    """
    def __init__(self, dim, depth, query_dim, key_dim, mlp_ratio=4, qkv_bias=False, drop_path=0., norm=None):
        super().__init__()
        norm = norm or nn.Identity
        self.norm1 = norm(dim)
        self.norm2 = norm(dim)

        layers = []
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        for i in range(depth):
            layers.append(
                ETL(
                    dim=dim,
                    query_dim=query_dim,
                    key_dim=key_dim,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=drop_path[i]
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.norm2(self.layer(self.norm1(x)))

        return x


class ETNet(nn.Module):
    """ Exemplar Transformer Network

    """
    def __init__(self, num_classes, embed_dim, depth, key_dim, mlp_ratio=4, qkv_bias=False, drop_path=0.1):
        super().__init__()

        self.patch_embed = ResNetBottleneck(1, embed_dim // 2)
        self.layer = ETB(dim=embed_dim, depth=depth, query_dim=embed_dim, key_dim=key_dim, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_path=drop_path, norm=LayerNorm)

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