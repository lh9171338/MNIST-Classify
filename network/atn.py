import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from network.resnet import ResNetBottleneck
from network.mhsa import LayerNorm
from network.ntn import FFN


class ASA(nn.Module):
    """ Axial Self-Attention

    """
    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=False, attn_drop=0., proj_drop=0., norm=None, size=None, groups=1, pos_encoding=True):
        super().__init__()
        assert not (pos_encoding and size is None), 'When using positional encoding, width can not be None'
        norm = norm or nn.Identity

        dim = dim * 2
        self.num_heads = num_heads
        self.pos_encoding = pos_encoding
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.avg = nn.AdaptiveAvgPool2d((1, None))
        self.query = nn.Conv2d(dim, dim, 1, bias=qkv_bias, groups=groups)
        self.key = nn.Conv2d(dim, dim, 1, bias=qkv_bias, groups=groups)
        self.value = nn.Conv2d(dim, dim, 1, bias=qkv_bias, groups=groups)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1, groups=groups)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=(1, sr_ratio), stride=(1, sr_ratio), groups=groups)
            self.norm = norm(dim)

        if pos_encoding:
            width, height = size
            assert width == height, 'Width must be equal to height'
            self.rel_w = nn.Parameter(torch.randn([1, num_heads, self.head_dim, width // sr_ratio]))

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.cat([x, x.transpose(-2, -1)], dim=1)

        x = self.avg(x)
        q = self.query(x).view(B, self.num_heads, self.head_dim, -1)
        if self.sr_ratio > 1:
            x = self.norm(self.sr(x))
        k = self.key(x).view(B, self.num_heads, self.head_dim, -1)
        v = self.value(x).view(B, self.num_heads, self.head_dim, -1)
        if self.pos_encoding:
            k = k + self.rel_w
        k = k * self.scale

        attn = q.transpose(-2, -1) @ k
        attn = self.attn_drop(self.softmax(attn))
        out = (v @ attn.transpose(-2, -1)).reshape(B, 2 * C, W).unsqueeze(dim=2)
        out = self.proj_drop(self.proj(out))
        out = out[:, :C] + out[:, C:].transpose(-2, -1)

        return out


class ATL(nn.Module):
    """ Axial Transformer Layer

    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4, sr_ratio=1, qkv_bias=False, drop=0., attn_drop=0., drop_path=0., norm=None, size=None, groups=1, pos_encoding=True):
        super().__init__()
        norm = norm or nn.Identity
        self.norm1 = norm(dim)
        self.norm2 = norm(dim)

        self.attn = ASA(dim=dim, num_heads=num_heads, sr_ratio=sr_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop,
                         proj_drop=drop, norm=norm, size=size, groups=groups, pos_encoding=pos_encoding)
        self.ffn = FFN(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x


class ATB(nn.Module):
    """ Axial Transformer Block

    """
    def __init__(self, dim, depth, num_heads=8, mlp_ratio=4, sr_ratio=1, qkv_bias=False, drop=0., attn_drop=0., drop_path=0., norm=None, size=None, groups=1, pos_encoding=True):
        super().__init__()
        norm = norm or nn.Identity
        self.norm1 = norm(dim)
        self.norm2 = norm(dim)

        layers = []
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        for i in range(depth):
            layers.append(
                ATL(
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
                    groups=groups,
                    pos_encoding=pos_encoding
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.norm2(self.layer(self.norm1(x)))

        return x


class ATNet(nn.Module):
    """ Axial Transformer Network

    """
    def __init__(self, num_classes, embed_dim, depth, num_heads=8, mlp_ratio=4, sr_ratio=1, qkv_bias=False, drop_path=0.1, groups=1, pos_encoding=True):
        super().__init__()

        self.patch_embed = ResNetBottleneck(1, embed_dim // 2)
        self.layer = ATB(dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, sr_ratio=sr_ratio,
                         qkv_bias=qkv_bias, drop_path=drop_path, norm=LayerNorm, size=(56, 56), groups=groups, pos_encoding=pos_encoding)

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
