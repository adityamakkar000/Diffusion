import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class DiTConfig:
    patch_size: int = MISSING
    hidden_size: int = MISSING
    layers: int = MISSING
    dropout: float = MISSING
    n_heads: int = MISSING

class TimeEmbedding(nn.Module):
    def __init__(self, T: int, t_emb: int):
        super().__init__()

        # assert t_emb % 2 == 0, "t_emb must be even"
        timesteps = torch.arange(T)
        exp = (2 / t_emb) * torch.arange(t_emb // 2)
        angluar_freq = torch.pow(1 / 10000, exp)
        theta = angluar_freq[..., None] * timesteps
        theta = theta.T
        sin = torch.sin(theta)  # (T, t_emb//2)
        cos = torch.cos(theta)  # (T, t_emb//2)

        self.pos_embedding = torch.stack([sin, cos], dim=-1).reshape(T, t_emb)

        self.register_buffer("pe", self.pos_embedding)

        self.ffn = nn.Sequential(
            nn.Linear(t_emb, t_emb),
            nn.SiLU(),
            nn.Linear(t_emb, t_emb),
        )

    def forward(self, t: Tensor):
        assert t.dim() == 1

        t_emb = self.pe[t]  # (b, t_emb)
        t_emb = self.ffn(t_emb)[:, None, ...]  # (b, 1,  4 * t_emb)

        return t_emb


class MHA(nn.Module):
    def __init__(self, hidden_size, heads, dropout):
        super().__init__()

        self.n_heads = heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // heads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.qkv(x)  # b x t x (3 hidden_size)
        x = rearrange(
            x, "b t (n h s) ->  n b h t s", n=3, h=self.n_heads, s=self.head_dim
        )
        q, k, v = torch.chunk(x, 3, dim=0)  # each: (b n_heads T head_size)
        wei = torch.einsum("...bnth, ...bnTh ->bntT", q, k) / self.hidden_size**0.5
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout1(wei)

        x = torch.einsum("bntT, ...bnTh ->bnth", wei, v)
        x = rearrange(x, "b n t h -> b t (n h)")
        x = self.dropout2(self.ffn(x))

        return x


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, n_heads,dropout):
        super().__init__()


        self.hidden_size = hidden_size
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)

        self.MHA = MHA(hidden_size, n_heads, dropout)

        self.scale_shift = nn.Sequential(
          nn.Linear(hidden_size, 6 * hidden_size),
          nn.SiLU(),
          nn.Linear(6 *hidden_size, 6 *hidden_size),
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:

        t_emb = self.scale_shift(t)
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = torch.chunk(
           t_emb, 6, dim=-1
        )

        residual = x
        x = gamma_1 * self.layernorm1(x) + beta_1
        x = self.MHA(x)
        x = alpha_1 * x
        x = residual + x

        residual = x
        x = gamma_2 * self.layernorm2(x) + beta_2
        x = self.ffn(x)
        x = alpha_2 * x
        x = residual + x

        return x

class DiT(nn.Module):
    def __init__(self, T, length, patch_size, hidden_size, layers, dropout, n_heads):

        super().__init__()

        assert (
            length % patch_size == 0
        ), f"length {length} must be divisible by patch_size {patch_size}"
        assert hidden_size % 2 == 0, f"hidden size {hidden_size} must be divisible by 2"
        self.length = length
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.T = T

        self.embedding = nn.Conv2d(
            3, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.time_embedding = TimeEmbedding(T, hidden_size)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, n_heads, dropout) for _ in range(layers)]
        )
        self.ln = nn.LayerNorm(self.hidden_size)
        self.ffn = nn.Linear(self.hidden_size, self.patch_size**2 * 3)

        T = (length // patch_size) ** 2
        timesteps = torch.arange(T)
        i = (2 / self.hidden_size) * torch.arange(self.hidden_size // 2)
        freq = torch.exp(-i * torch.log(torch.Tensor([10000.0])))
        values = timesteps[..., None] * freq  # T x hidden_size//2
        pos_emb = rearrange(
            torch.stack([values.sin(), values.cos()], dim=-1), "t d c -> t (d c)"
        )

        self.register_buffer("pos_emb", pos_emb, persistent=False)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = self.embedding(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x += self.pos_emb
        t = self.time_embedding(t)
        for block in self.blocks:
            x = block(x, t)
        x = self.ffn(self.ln(x))

        x = rearrange(
            x,
            "b (h1 w1) (p1 p2 c) -> b c (h1 p1) (w1 p2) ",
            h1=h // self.patch_size,
            w1=w // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=3,
        )

        return x
