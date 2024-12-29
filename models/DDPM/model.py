import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List
from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class UNetConfig:
    T: int = MISSING
    resNetBlocks: int = MISSING
    attn: List[int] = MISSING
    dropout: float = MISSING
    ch: int = MISSING
    ch_mult: List[int] = MISSING
    groupSize: int = 8


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

        self.pos = nn.Embedding(T, t_emb)
        self.ffn = nn.Sequential(
            nn.Linear(t_emb, 4 * t_emb),
            nn.SiLU(),
            nn.Linear(4 * t_emb, 4 *t_emb),
        )

    def forward(self, t: Tensor):
        assert t.dim() == 1

        t_emb = self.pe[t] # (b, t_emb)
        t_emb = self.ffn(t_emb) # (b, 4 * t_emb)

        return t_emb


class ResNetBlock(nn.Module):
    def __init__(
        self, ch:int, numGroups: int, inChannels: int, outChannels: int, dropout: float
    ) -> None:
        super().__init__()

        assert outChannels % numGroups == 0

        self.ch = inChannels

        self.gn1 = nn.Sequential(
            nn.GroupNorm(numGroups, inChannels),
            nn.SiLU(),
            nn.Conv2d(inChannels, outChannels, 3, 1, 1),
        )

        self.t_ffn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(4 * ch, outChannels),
        )

        self.gn2 = nn.Sequential(
            nn.GroupNorm(numGroups, outChannels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(outChannels, outChannels, 3, 1, 1),
        )

        if inChannels != outChannels:
            self.conv1x1 = nn.Conv2d(inChannels, outChannels, 1, 1, 0)
        else:
            self.conv1x1 = nn.Identity()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        assert x.dim() == 4
        assert x.shape[1] == self.ch

        residual = self.conv1x1(x)

        x = self.gn1(x)
        x += self.t_ffn(t)[..., None, None]

        assert x.shape == residual.shape

        x = self.gn2(x) + residual

        return x


class Attention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        in_channel: int,
    ) -> None:
        super().__init__()
        assert in_channel % n_heads == 0
        self.qkv = nn.Conv2d(
            in_channel, 3 * in_channel, kernel_size=1, stride=1, padding=0
        )
        self.proj = nn.Conv2d(
            in_channel, in_channel, kernel_size=1, stride=1, padding=0
        )
        self.n_heads = n_heads
        self.emb = in_channel // n_heads

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(b, 3, self.n_heads, self.emb, h * w)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q,k,v = qkv.chunk(3, dim=0) # Each: [B, n_heads, H * W, emb]

        attn = torch.einsum('...bhte,...bhTe->bhtT', q, k) * (self.emb ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bhtT, ...bhTe -> bhte", attn, v) #
        out = out.transpose(-2, -1)
        out = out.reshape(b, c, h, w)

        return self.proj(out)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int, n_heads: int, dropout: float) -> None:
        super().__init__()

        self.gn = nn.GroupNorm(n_heads, in_channels)
        self.attn = Attention(n_heads, in_channels)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        residual = x

        x = self.gn(x)
        x = self.attn(x)

        return x + residual


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv2d(channels, channels, 3, 2, 1)

    def forward(self, x: Tensor, t:Tensor) -> Tensor:
        downsample = self.downsample(x)
        assert (
            downsample.shape[2] == x.shape[2] // 2
            and downsample.shape[3] == x.shape[3] // 2
        )
        return downsample


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        upsample = F.interpolate(x, scale_factor=2)
        upsample = self.conv(upsample)
        return upsample


class UNET(nn.Module):
    def __init__(
        self,
        T: int,
        resNetBlocks: int,
        attn: List[int],
        dropout: float,
        ch,
        ch_mult: List[int],
        groupSize: int = 8,
    ) -> None:
        super().__init__()

        assert len(ch_mult) + 1 == len(attn) and attn[-1] > 0, "attn not provided correctly"

        self.time_embedding = TimeEmbedding(T, ch)
        self.conv_in = nn.Conv2d(3, ch, 3, 1, 1)

        self.end = nn.Sequential(
            nn.GroupNorm(groupSize, ch), nn.SiLU(), nn.Conv2d(ch, 3, 3, 1, 1)
        )

        self.downBlocks = nn.ModuleList()
        self.upBlocks = nn.ModuleList()

        num_of_blocks = len(ch_mult)
        chs = [ch]
        current_channels = ch

        for i in range(num_of_blocks):
            out_channels = ch * ch_mult[i]
            for _ in range(resNetBlocks):
                self.downBlocks.append(
                    ResNetBlock(ch, groupSize, current_channels, out_channels, dropout)
                )
                chs.append(out_channels)
                current_channels = out_channels
                if attn[i] > 0:
                    print('hello')
                    self.downBlocks.append(
                        AttentionBlock(current_channels, attn[i], dropout)
                    )
            if i < num_of_blocks - 1:
                self.downBlocks.append(Downsample(current_channels))
                chs.append(out_channels)

        self.middleBlocks = nn.ModuleList(
            [
                ResNetBlock(ch, groupSize, current_channels, current_channels, dropout),
                AttentionBlock(current_channels, attn[-1], dropout),
                ResNetBlock(ch, groupSize, current_channels, current_channels, dropout),
            ]
        )

        for i in reversed(range(num_of_blocks)):
            out_channels = ch * ch_mult[i]

            for _ in range(resNetBlocks + 1):
                self.upBlocks.append(
                    ResNetBlock(ch, groupSize, current_channels + chs[-1], out_channels, dropout)
                )
                chs.pop()
                current_channels = out_channels
                if attn[i] > 0:
                    self.upBlocks.append(AttentionBlock(out_channels, attn[i], dropout))
            if i > 0:
                self.upBlocks.append(Upsample(current_channels))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        assert x.dim() == 4
        assert x.shape[1] == 3

        x = self.conv_in(x)
        t = self.time_embedding(t)


        down_outputs = [x]
        for block in self.downBlocks:
            x = block(x, t)
            if isinstance(block, AttentionBlock):
                down_outputs.pop()
            down_outputs.append(x)

        for block in self.middleBlocks:
            x = block(x, t)

        for block in self.upBlocks:
            if isinstance(block, ResNetBlock):
                x = torch.cat([x, down_outputs[-1]], dim=1)  # along channels
                down_outputs.pop()

            x = block(x, t)
        x = self.end(x)

        return x
