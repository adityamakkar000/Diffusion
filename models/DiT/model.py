import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange


class ScaleShift(nn.Module):

  def __init__(self, hidden_size):
    super().__init__()

    self.shift = nn.Parameter(torch.zeros((hidden_size,)), requires_grad=True)
    self.scale = nn.Parameter(torch.randn(hidden_size), requires_grad=True)

  def forward(self, x: Tensor) -> Tensor:
    return x * self.scale + self.shift

class Shift(nn.Module):

  def __init__(self, hidden_size):
    super().__init__()

    self.shift = nn.Parameter(torch.zeros((hidden_size,)), requires_grad=True)

  def forward(self, x: Tensor) -> Tensor:
    return x + self.shift


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
      nn.Linear(4 * hidden_size, hidden_size)
    )

  def forward(self, x: Tensor) -> Tensor:

    x = self.qkv(x) # b x t x (3 hidden_size)
    x = rearrange(x, 'b t (n h s) ->  n b t h s', n=3, h=self.n_heads, s=self.head_dim)
    q,k,v = torch.chunk(x, 3, dim=2) # each: (b n_heads T head_size)

    print(q.shape, k.shape, v.shape)
    wei = torch.einsum('...bnth, ...bnTh ->bntT', q, k) /  self.hidden_size**0.5
    wei = torch.softmax(wei,dim=-1)
    wei = self.dropout1(wei)

    print(wei.shape, v.shape)
    x = torch.einsum('bntT, ...bnTh ->bnth', wei, v)
    x = rearrange(x, 'b n t h -> b t (n h)')
    x = self.dropout2(self.ffn(x))

    return x

class DiTBlock(nn.Module):

  def __init__(self, hidden_size, n_heads, dropout):
    super().__init__()

    self.hidden_size = hidden_size
    self.layernorm1 = nn.LayerNorm(hidden_size)
    self.layernorm2 = nn.LayerNorm(hidden_size)

    self.MHA = MHA(hidden_size, n_heads, dropout)

    self.scale_shift1 = ScaleShift(hidden_size)
    self.scale_shift2 = ScaleShift(hidden_size)

    self.shift1 = Shift(hidden_size)
    self.shift2 = Shift(hidden_size)

    self.ffn = nn.Sequential(
      nn.Linear(hidden_size, 4 * hidden_size),
      nn.GELU(),
      nn.Linear(4 * hidden_size, hidden_size)
    )

  def forward(self, x: Tensor) -> Tensor:

    residual = x
    x = self.scale_shift1(self.layernorm1(x))
    x = self.MHA(x)
    x = self.shift1(x)
    x = residual + x

    residual = x
    x = self.scale_shift2(self.layernorm2(x))
    x = self.ffn(x)
    x = self.shift2(x)
    x = residual + x

    return x

class DiT(nn.Module):

  def __init__(self, length, patch_size, hidden_size, layers, dropout, n_heads):

    super().__init__()

    assert length % patch_size == 0, f"length {length} must be divisible by patch_size {patch_size}"
    assert hidden_size % 2 == 0, f"hidden size {hidden_size} must be divisible by 2"
    self.length = length
    self.patch_size = patch_size
    self.hidden_size = hidden_size

    self.embedding = nn.Conv2d(3, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
    self.blocks = nn.ModuleList([
      DiTBlock(hidden_size, n_heads, dropout) for _ in range(layers)
    ])
    self.ln = nn.LayerNorm(self.hidden_size)
    self.ffn = nn.Linear(self.hidden_size, self.patch_size**2 * 3)

    T = (length // patch_size) ** 2
    timesteps = torch.arange(T)
    i = (2/self.hidden_size) * torch.arange(self.hidden_size//2)
    freq = torch.exp(-i * torch.log(torch.Tensor([10000.0])))
    values = timesteps[..., None] * freq # T x hidden_size//2
    pos_emb = rearrange(torch.stack([values.sin(), values.cos()], dim=-1), 't d c -> t (d c)')

    self.register_buffer('pos_emb', pos_emb, persistent=False)

  def forward(self, x: Tensor) -> Tensor:

    print(x.shape)
    x = self.embedding(x)
    print(x.shape)
    x = rearrange(x, 'b c h w -> b (h w) c')
    x += self.pos_emb

    for block in self.blocks:
      x = block(x)
    x = self.ffn(self.ln(x))

    return x


if __name__ == '__main__':

  config = {
    'length': 32,
    'patch_size': 4,
    'hidden_size': 64,
    'layers': 2,
    'dropout': 0.1,
    'n_heads': 8
  }
  model = DiT(256, 16, 512, 6, 0.1, 8)
  print(sum(p.numel() for p in model.parameters() if p.requires_grad))
  x = torch.randn(1, 3, 32, 32)
  y = model(x)
  # print(y.shape)
