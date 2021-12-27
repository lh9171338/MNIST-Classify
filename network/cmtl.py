from torch import nn


class MFFN(nn.Module):
    """ Mix-FFN

    """
    def __init__(self, dim, mlp_ratio=1):
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


class SRA(nn.Module):
    """ Spatial-Reduction Attention

    """
    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.query = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.key = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio, bias=qkv_bias)
        self.value = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.query(x).reshape(B, self.num_heads, C // self.num_heads, -1)
        k = self.key(x).reshape(B, self.num_heads, C // self.num_heads, -1)
        v = self.value(x).reshape(B, self.num_heads, C // self.num_heads, -1)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        out = self.proj(out)
        return out


class CMTL(nn.Module):
    """ Convolutional Mix Transformer Layer

    """
    def __init__(self, dim, num_heads, mlp_ratio=4, sr_ratio=1, qkv_bias=False):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.norm2 = nn.GroupNorm(1, dim)

        self.attn = SRA(dim, num_heads, sr_ratio, qkv_bias)
        self.mffn = MFFN(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.norm1(self.attn(x))
        x = x + self.norm2(self.mffn(x))

        return x


class CMTLNet(nn.Module):
    def __init__(self, in_dim, num_classes, depth, embed_dim, num_heads, mlp_ratio, sr_ratio, qkv_bias):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.append(
                CMTL(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    sr_ratio=sr_ratio,
                    qkv_bias=qkv_bias
                )
            )

        self.patch_embed = nn.Conv2d(in_dim, embed_dim, 3, stride=2, padding=1)
        self.layer = nn.Sequential(*layers)
        self.head = nn.Linear(embed_dim, num_classes)

        self.norm1 = nn.GroupNorm(1, embed_dim)
        self.norm2 = nn.GroupNorm(1, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.norm2(self.layer(self.norm1(x)))
        x = x.mean(-1).mean(-1)
        x = self.head(x)
        x = self.softmax(x)

        return x