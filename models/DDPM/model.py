import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import List, Tuple, Optional, Union


class ResNetBlock(nn.Module):
    def __init__(self, numGroups: int, inChannels: int, outChannels: int) -> None:
        super().__init__()

        assert outChannels % numGroups == 0

        self.ch = inChannels

        self.network = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, 3, 1, 1),
            nn.GroupNorm(numGroups, outChannels),
            nn.ReLU(),
            nn.Conv2d(outChannels, outChannels, 3, 1, 1),
            nn.GroupNorm(numGroups, outChannels),
        )

        self.conv1x1 = nn.Conv2d(inChannels, outChannels, 1, 1, 0)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 4
        assert x.shape[1] == self.ch

        x_forward = self.network(x)
        y = self.conv1x1(x)
        logits = self.relu(x_forward + y)

        return logits


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        timesteps: int,
        size: Tuple[int, int],
    ):
        super().__init__()

        self.pos_embedding = nn.Embedding(timesteps, size[0] * size[1])
        self.size = size

    def forward(self, t: Tensor):
        assert t.dim() == 1

        b = t.shape[0]
        t_emb = self.pos_embedding(t)
        t_emb = t_emb.view(b, 1, self.size[0], self.size[1])

        return t_emb


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
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.emb = n_emb // n_heads  # embedding per head

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 4

        b, c, h, w = x.shape
        qkv = self.qkv(x)  # pass through projection
        qkv = qkv.view(b, 3 * self.n_heads, c // self.n_heads, h * w).transpose(-2, -1)
        q, k, v = qkv.split(self.n_heads, dim=1)
        attn = F.softmax((q @ k.transpose(2, 3)) / (self.emb**0.5), dim=-1)
        attn = self.dropout(attn)
        self.logits = attn @ v  # (b, hs, h * w, c // hs)
        self.logits = self.logits.transpose(1, 2).view(b, h * w, c)
        self.logits = self.logits.permute(0, 2, 1)  # reshape back into images
        self.logits = self.logits.view(b, c, h, w)
        return self.dropout(self.proj(self.logits))


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

