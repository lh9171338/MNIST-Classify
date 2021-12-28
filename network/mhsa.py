import torch
import torch.nn as nn
import torch.nn.functional as F


class MHSA(nn.Module):
    """ Multi-Head Self-Attention

    """
    def __init__(self, dim, num_heads=4, sr_ratio=1, qkv_bias=False, width=None, height=None, pos_encoding=False):
        super().__init__()
        self.num_heads = num_heads
        self.pos_encoding = pos_encoding
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.query = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.key = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio, bias=qkv_bias)
        self.value = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio, bias=qkv_bias)

        if pos_encoding:
            self.rel_h = nn.Parameter(torch.randn([1, num_heads, head_dim, 1, height // sr_ratio]), requires_grad=True)
            self.rel_w = nn.Parameter(torch.randn([1, num_heads, head_dim, width // sr_ratio, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, self.num_heads, C // self.num_heads, -1)
        k = self.key(x).view(B, self.num_heads, C // self.num_heads, -1)
        v = self.value(x).view(B, self.num_heads, C // self.num_heads, -1)

        attn = torch.matmul(q.transpose(-2, -1), k)

        if self.pos_encoding:
            pos = (self.rel_h + self.rel_w).view(1, self.num_heads, C // self.num_heads, -1)
            pos = torch.matmul(q.transpose(-2, -1), pos)
            attn += pos
        attn = self.softmax(attn * self.scale)
        out = torch.matmul(v, attn.transpose(-2, -1)).view(B, C, H, W)

        return out


class MHSABottleneck(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, qkv_bias=False, width=None, height=None, pos_encoding=False, dim_factor=2, scale_factor=1):
        super().__init__()
        hidden_dim = dim // dim_factor
        self.scale_factor = scale_factor
        sr_ratio = sr_ratio // scale_factor
        width = width // scale_factor
        height = height // scale_factor

        self.bn1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, hidden_dim, 1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.mhsa = MHSA(hidden_dim, num_heads=num_heads, sr_ratio=sr_ratio, qkv_bias=qkv_bias, width=width, height=height, pos_encoding=pos_encoding)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, dim, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        x = self.conv1(self.relu(self.bn1(x)))
        if self.scale_factor > 1:
            x = F.max_pool2d(x, self.scale_factor, stride=self.scale_factor)
        x = self.mhsa(self.relu(self.bn2(x)))
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor)
        x = self.conv3(self.relu(self.bn3(x)))

        x += shortcut

        return x


class MHSANet(nn.Module):
    def __init__(self, in_dim, num_classes, depth, embed_dim, num_heads=8, sr_ratio=1, qkv_bias=False, width=None, height=None, pos_encoding=False):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.append(
                MHSABottleneck(
                    dim=embed_dim,
                    num_heads=num_heads,
                    sr_ratio=sr_ratio,
                    qkv_bias=qkv_bias,
                    width=width,
                    height=height,
                    pos_encoding=pos_encoding
                )
            )

        self.patch_embed = nn.Conv2d(in_dim, embed_dim, 3, stride=2, padding=1)
        self.layer = nn.Sequential(*layers)
        self.head = nn.Linear(embed_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.layer(x)
        x = x.mean(-1).mean(-1)
        x = self.head(x)
        x = self.softmax(x)

        return x