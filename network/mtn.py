import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from network.resnet import ResNetBottleneck


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=1, drop=0.):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.drop(self.act(x))
        x = self.drop(self.fc2(x))
        return x


class SRA(nn.Module):
    """ Spatial-Reduction Attention

    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, sr_ratio=1, attn_drop=0., proj_drop=0., norm=None, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        norm = norm or nn.Identity
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
                self.norm = norm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, 1)
            self.norm = norm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.query(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        if not self.linear:
            if self.sr_ratio > 1:
                x = x.transpose(1, 2).reshape(B, C, H, W)
                x = self.norm(self.sr(x).flatten(2).transpose(1, 2))
        else:
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.act(self.norm(self.sr(self.pool(x)).flatten(2).transpose(1, 2)))
        k = self.key(x).reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.value(x).reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MTL(nn.Module):
    """ Mix Transformer Layer

    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.1, sr_ratio=1,
                 norm=None, linear=False):
        super().__init__()
        norm = norm or nn.Identity
        self.norm1 = norm(dim)
        self.norm2 = norm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = SRA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                        sr_ratio=sr_ratio, norm=norm, linear=linear)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class MTB(nn.Module):
    """ Mix Transformer Block

    """
    def __init__(self, dim, depth, num_heads=8, mlp_ratio=4, sr_ratio=1, qkv_bias=False, drop=0., attn_drop=0., drop_path=0., norm=None, linear=False):
        super().__init__()
        norm = norm or nn.Identity
        self.norm1 = norm(dim)
        self.norm2 = norm(dim)

        layers = []
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        for i in range(depth):
            layers.append(
                MTL(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    sr_ratio=sr_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm=norm,
                    linear=linear
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)

        for layer in self.layers:
            x = layer(x, H, W)

        x = self.norm2(x)
        x = x.transpose(1, 2).view(B, C, H, W)

        return x


class MTNet(nn.Module):
    """ Mix Transformer Network

    """
    def __init__(self, num_classes, embed_dim, depth, num_heads=8, mlp_ratio=4, sr_ratio=1, qkv_bias=False, drop_path=0.1, linear=False):
        super().__init__()

        self.patch_embed = ResNetBottleneck(1, embed_dim // 2)
        self.layer = MTB(dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, sr_ratio=sr_ratio,
                         qkv_bias=qkv_bias, drop_path=drop_path, norm=nn.LayerNorm, linear=linear)

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
