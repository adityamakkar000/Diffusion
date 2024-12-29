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

class Sample(nn.Module):
    def __init__(
        self,
        inChannel: int,
        outChannel: int,
        numGroups: int,
        stride: Optional[int],
        dropout: Optional[float],
        n_heads: Union[int, None],
        resNetBlocks: int,
        upsample: bool,
        sample: bool,
    ):
        super().__init__()

        self.network = nn.ModuleList(
            [
                ResNetBlock(
                    numGroups,
                    inChannel if i == 0 else outChannel,
                    outChannel,
                )
                for i in range(resNetBlocks)
            ]
        )

        if n_heads is not None:
            self.network.append(Attention(n_heads, outChannel, dropout))

        self.sampleBool = (sample, upsample)

        if sample:
            if not upsample:
                self.sample = nn.Conv2d(outChannel, outChannel, 3, stride, 1)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.network:
            x = block(x)

        if self.sampleBool[0] and not self.sampleBool[1]:
            x = self.sample(x)

        return x


class UNET(nn.Module):
    def __init__(
        self,
        timeStep: int,
        originalSize: Tuple[int, int],
        inChannels: int,
        channels: List[int],
        strides: List[int],
        n_heads: List[int],
        resNetBlocks: List[int],
        attn: List[bool],
        dropout: List[float],
    ) -> None:
        super().__init__()

        assert len(n_heads) == sum(attn)
        assert len(channels) == len(strides) + 1 == len(attn) == len(resNetBlocks)
        self.T = timeStep
        channels = [inChannels] + channels  # for time embedding
        length = len(channels)
        n_heads_downsample = n_heads.copy()
        n_heads_upsample = n_heads.copy()
        dropout_downsample = dropout.copy()
        dropout_upsample = dropout.copy()

        n_heads_downsample.reverse()
        dropout_downsample.reverse()
        dummy = 1
        strides.append(dummy)  # make same length doens't get used

        # start
        self.time_emb = TimeEmbedding(timeStep, originalSize)
        self.conv_in = nn.Conv2d(inChannels, inChannels, 3, 1, 1)

        # downsample layers
        self.downsample_layers = nn.ModuleList(
            [
                Sample(
                    channels[i],
                    channels[i + 1],
                    1,
                    strides[i],
                    dropout_downsample.pop() if attn[i] else None,
                    n_heads_downsample.pop() if attn[i] else None,
                    resNetBlocks[i],
                    upsample=False,
                    sample=True if i < length - 1 else False,
                )
                for i in range(length - 1)
            ]
        )

        # upsample layers
        self.upsample_layers = nn.ModuleList(
            [
                Sample(
                    channels[i] + (0 if i == length - 1 else channels[i]),
                    channels[i - 1],
                    1,
                    strides[i - 1],
                    dropout_upsample.pop() if attn[i - 1] else None,
                    n_heads_upsample.pop() if attn[i - 1] else None,
                    resNetBlocks[i - 1],
                    upsample=True,
                    sample=True if i < length - 1 else False,
                )
                for i in range(length - 1, 0, -1)
            ]
        )

        # end
        self.end = ResNetBlock(1, 2 * channels[0], inChannels)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        assert t.shape[0] == x.shape[0]

        time_emb = self.time_emb(t)
        x = self.conv_in(x) + time_emb  # (b, c, h, w) + (1,1,h,w)

        skip_connections = [x]

        for block in self.downsample_layers:
            x = block(x)
            skip_connections.append(x)

        for i, block in enumerate(self.upsample_layers):
            sc = skip_connections.pop()
            if i > 0:
                x = torch.cat([F.interpolate(x, (sc.shape[2], sc.shape[3])), sc], dim=1)
            x = block(x)

        sc = skip_connections.pop()
        x = torch.cat([F.interpolate(x, (sc.shape[2], sc.shape[3])), sc], dim=1)

        x = self.end(x)

        return x
