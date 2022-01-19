import torch
import torch.nn as nn
import torch.nn.functional as F
from network.resnet import ResNetBottleneck


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class MHSA(nn.Module):
    """ Multi-Head Self-Attention

    """
    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=False, attn_drop=0., proj_drop=0., norm=None, size=None, pos_encoding=False):
        super().__init__()
        assert not (pos_encoding and size is None), 'When using positional encoding, size can not be None'
        norm = norm or nn.Identity

        self.num_heads = num_heads
        self.pos_encoding = pos_encoding
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.query = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.key = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.value = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = norm(dim)

        if pos_encoding:
            width, height = size
            self.rel_h = nn.Parameter(torch.randn([1, num_heads, head_dim, 1, height // sr_ratio]))
            self.rel_w = nn.Parameter(torch.randn([1, num_heads, head_dim, width // sr_ratio, 1]))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, self.num_heads, C // self.num_heads, -1)
        if self.sr_ratio > 1:
            x = self.norm(self.sr(x))
        k = self.key(x).view(B, self.num_heads, C // self.num_heads, -1)
        v = self.value(x).view(B, self.num_heads, C // self.num_heads, -1)
        if self.pos_encoding:
            pos = (self.rel_h + self.rel_w).view(1, self.num_heads, C // self.num_heads, -1)
            k = k + pos
        k = k * self.scale

        attn = q.transpose(-2, -1) @ k
        attn = self.attn_drop(self.softmax(attn))
        out = (v @ attn.transpose(-2, -1)).view(B, C, H, W)
        out = self.proj_drop(self.proj(out))

        return out


class MHSABottleneck(nn.Module):
    expansion = 2

    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=False, size=None, pos_encoding=False):
        super().__init__()
        hidden_dim = dim // self.expansion

        self.bn1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, hidden_dim, 1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.mhsa = MHSA(hidden_dim, num_heads=num_heads, sr_ratio=sr_ratio, qkv_bias=qkv_bias, norm=LayerNorm, size=size, pos_encoding=pos_encoding)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, dim, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.mhsa(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        out += x
        return out


class MHSANet(nn.Module):
    def __init__(self, num_classes, embed_dim, depth, num_heads=8, sr_ratio=1, qkv_bias=False, pos_encoding=False):
        super().__init__()

        self.patch_embed = ResNetBottleneck(1, embed_dim // 2)
        layers = [MHSABottleneck(dim=embed_dim, num_heads=num_heads, sr_ratio=sr_ratio, qkv_bias=qkv_bias, size=(56, 56), pos_encoding=pos_encoding) for _ in range(depth)]
        self.layer = nn.Sequential(*layers)

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