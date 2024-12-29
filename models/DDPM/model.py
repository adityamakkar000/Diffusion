import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class UNetConfig:
    timeStep: int = MISSING
    originalSize: Tuple[int, int] = MISSING
    inChannels: int = MISSING
    channels: List[int] = MISSING
    strides: List[int] = MISSING
    n_heads: List[int] = MISSING
    resNetBlocks: List[int] = MISSING
    attn: List[bool] = MISSING
    dropout: List[float] = MISSING



class TimeEmbedding(nn.Module):
    def __init__(
        self,
        T: int,
        t_emb: int
    ):
        super().__init__()

        assert t_emb % 2 == 0, "t_emb must be even"
        timesteps = torch.arange(T)
        exp = (2/t_emb) * torch.arange(t_emb//2)
        angluar_freq = torch.pow(1/10000, exp)
        theta = angluar_freq[..., None] * timesteps
        theta = theta.T
        sin = torch.sin(theta) # (T, t_emb//2)
        cos = torch.cos(theta) # (T, t_emb//2)

        self.pos_embedding = torch.stack([sin, cos],dim=-1).view(T, t_emb)
        self.ffn = nn.Sequential(
            nn.Linear(t_emb, 4 * t_emb),
            nn.SiLU(),
            nn.Linear(4 * t_emb, 4 * t_emb),
        )

    def forward(self, t: Tensor):
        assert t.dim() == 1

        t_emb = self.pos_embedding(t) # (b, t_emb)
        t_emb = self.ffn(t_emb)[..., None, None]# (b, 4 * t_emb, 1, 1)

        return t_emb

class ResNetBlock(nn.Module):
    def __init__(self, numGroups: int, inChannels: int, outChannels: int, dropout: float) -> None:
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
            nn.LazyLinear(outChannels),
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
        x += self.t_ffn(t)

        assert x.shape == residual.shape

        x = self.gn2(x) + residual

        return x

class Attention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        in_channel: int,
        dropout: float,
    ) -> None:
        super().__init__()

        n_emb = in_channel
        assert n_emb % n_heads == 0

        self.qkv = nn.Conv2d(
            in_channel, 3 * in_channel, kernel_size=1, stride=1, padding=0
        )  # combine qkv for efficency
        self.proj = nn.Conv2d(
            in_channel, in_channel, kernel_size=1, stride=1, padding=0
        )

        self.n_heads = n_heads
        self.emb = n_emb // n_heads  # embedding per head

    def forward(self, x: Tensor) -> Tensor:

        assert x.dim() == 4

        b, c, h, w = x.shape
        qkv = self.qkv(x) # (b, 3 * c, h, w)
        qkv = qkv.view(b, 3 * c, h * w)
        qkv = qkv.permute(0, 2, 1) # (b, h * w, 3 * c)
        qkv = qkv.view(b, h * w, 3 * self.n_heads, self.emb)
        q, k, v = qkv.chunk(3, dim=2) # (b, h * w, n_heads, emb) x 3

        w = torch.einsum("bthe, bThe -> bhtT", q, k) / (self.emb ** 0.5)
        w = F.softmax(w, dim=-1)

        logits = torch.einsum("bhtT, bThe -> bthe", w, v)
        logits = logits.view(b, h * w, self.n_heads * self.emb)
        logits = logits.permute(0, 2, 1).view(b, c, h, w)
        logits = self.proj(logits)

        return logits

class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int, n_heads: int, dropout: float) -> None:
        super().__init__()

        self.gn = nn.GroupNorm(n_heads, in_channels)
        self.attn = Attention(n_heads, in_channels, dropout)

    def forward(self,x: Tensor) -> Tensor:
        residual = x

        x = self.gn(x)
        x = self.attn(x)

        return x + residual

class Downsample(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv2d(channels, channels, 3, 2, 1)

    def forward(self, x: Tensor) -> Tensor:

        downsample = self.downsample(x)
        assert downsample.shape[2] == x.shape[2] // 2 and downsample.shape[3] == x.shape[3] // 2
        return downsample

class Upsample(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
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
        groupSize: int = 8
    ) -> None:

        super().__init__()

        self.time_embedding = TimeEmbedding(T, ch)
        self.conv_in = nn.Conv2d(3, ch, 3, 1, 1)

        self.end = nn.Sequential(
            nn.GroupNorm(groupSize, ch),
            nn.SiLU(),
            nn.Conv2d(ch, 3, 3, 1,1)
        )

        self.downBlocks = nn.ModuleList()
        self.upBlocks = nn.ModuleList()

        num_of_blocks = len(ch_mult)
        chs = []
        current_channels = ch
        for i in range(num_of_blocks):
            out_channels = ch * ch_mult[i]
            for _ in range(resNetBlocks):
                self.downBlocks.append(ResNetBlock(groupSize, current_channels, out_channels, dropout))
                chs.append(current_channels)
                current_channels = out_channels
                if attn[i] > 0:
                    self.downBlocks.append(AttentionBlock(current_channels, attn[i], dropout))
            if i < num_of_blocks - 1:
                self.downBlocks.append(Downsample(current_channels))

        self.middleBlocks = nn.ModuleList(
            [
                ResNetBlock(groupSize, current_channels, current_channels, dropout),
                AttentionBlock(current_channels, attn[-1], dropout),
                ResNetBlock(groupSize, current_channels, current_channels, dropout),
            ])

        for i in range(reversed(num_of_blocks)):
            out_channels = ch * ch_mult[i]

            # +1 resnetBlocks since we have resNetBlocks+1 residual for each layer
            for _ in range(resNetBlocks + 1 ):
                self.upBlocks.append(ResNetBlock(groupSize, out_channels, chs[-1], dropout))
                chs.pop()
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
            down_outputs.append(x)

        for block in self.middleBlocks:
            x = block(x, t)

        for block in self.upBlocks:
            x = torch.cat([x, down_outputs[-1]], dim=1) # along channels
            down_outputs.pop()
           
            x = block(x, t)

        x = self.end(x)
        return x