import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from typing import List, Tuple, Optional, Union


class ResNetBlock(nn.Module):

    def __init__(self,
                 numGroups: int,
                 inChannels: int,
                 outChannels: int
                 ) -> None:

        super().__init__()


        assert outChannels % numGroups == 0

        self.ch = inChannels

        self.network = nn.Sequential(
                        nn.Conv2d(inChannels, outChannels, 3, 1,1),
                        nn.GroupNorm(numGroups, outChannels),
                        nn.ReLU(),
                        nn.Conv2d(outChannels, outChannels, 3,1,1),
                        nn.GroupNorm(numGroups, outChannels)
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

    def __init__(self,
                timesteps: int,
                size: Tuple[int,int],

                 ):

        super().__init__()

        self.pos_embedding = nn.Embedding(timesteps, size[0] * size[1])
        self.size = size

    def forward(self, t:Tensor):


        t_emb = self.pos_embedding(t)
        t_emb = t_emb.view(t_emb.shape[0], self.size[0], self.size[1]).unsqueeze(dim=1)

        assert t_emb.dim() == 4 and t_emb.shape[1] == 1

        return t_emb

class Attention(nn.Module):

    def __init__(self,
                 n_heads: int,
                 in_channel: int,
                 dropout: float,
                 ) -> None:

        super().__init__()

        n_emb = in_channel
        assert n_emb % n_heads == 0

        self.qkv = nn.Conv2d(in_channel, 3 * in_channel,kernel_size=1, stride=1, padding=0) # combine qkv for efficency
        self.proj = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.emb = n_emb // n_heads # embedding per head

    def forward(self, x: Tensor) -> Tensor:

        assert x.dim() == 4

        b,c,h,w = x.shape
        # pass through projection
        qkv = self.qkv(x)
        qkv = qkv.view(b,3 * self.n_heads, c // self.n_heads, h * w)
        qkv = qkv.transpose(-2, -1) # switch channels with time steps ( height and width)
        q,k,v = qkv.split(self.n_heads, dim=1)
        k= k.transpose(2,3)
        attn = F.softmax((q @ k)/(self.emb ** 0.5) , dim=-1)
        attn = self.dropout(attn)
        self.logits = (attn @ v) # (b, hs, h * w, c // hs)
        self.logits = self.logits.transpose(1,2).view(b, h * w, c)
        self.logits = self.logits.permute(0,2,1) # reshape back into images
        self.logits = self.logits.view(b, c, h,w )
        return self.dropout(self.proj(self.logits))



class Sample(nn.Module):

    def __init__(self,
                inChannel: int,
                outChannel: int,
                numGroups: int,
                stride: Optional[int],
                dropout: Optional[float],
                n_heads: Union[int, None],
                resNetBlocks: int,
                upsample: bool,
                sample: bool
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
            self.network.append(Attention(
                n_heads,
                outChannel,
                dropout
            ))

        self.sampleBool = (sample, upsample)


        if sample:
            if not upsample:
                self.sample = nn.Conv2d(outChannel, outChannel, 3, stride, 1)


    def forward(self,
                x: Tensor
                ) -> Tensor:

        for block in self.network:
            x = block(x)

        if self.sampleBool[0] and not self.sampleBool[1]:
            x = self.sample(x)

        return x



class UNET(nn.Module):

    def __init__(self,
                 timeStep: int,
                 orginalSize: Tuple[int,int],
                 inChannels: int,
                 channels: List[int],
                 strides: List[int],
                 n_heads: List[int],
                 resNetBlocks: List[int],
                 attn: List[bool],
                 dropout: List[float],
                 ) -> None :

        super().__init__()

        assert len(n_heads) == sum(attn)
        assert len(channels) == len(strides) + 1 == len(attn) == len(resNetBlocks)

        channels = [inChannels + 1] + channels # for time embedding
        length = len(channels)
        n_heads_downsample = n_heads.copy()
        n_heads_upsample = n_heads.copy()
        dropout_downsample = dropout.copy()
        dropout_upsample = dropout.copy()

        n_heads_downsample.reverse()
        dropout_downsample.reverse()
        dummy=1
        strides.append(dummy) # make same length doens't get used

        self.time_emb = TimeEmbedding(timeStep, orginalSize)

        self.downsample_layers = nn.ModuleList(
            [
                Sample(
                    channels[i],
                    channels[i+1],
                    1,
                    strides[i],
                    dropout_downsample.pop() if attn[i] else None,
                    n_heads_downsample.pop() if attn[i] else None,
                    resNetBlocks[i],
                    upsample=False,
                    sample=True if i < length - 1 else False
                ) for i in range(length - 1)
            ]
                                               )



        self.upsample_layers = nn.ModuleList(
            [
                Sample(
                    channels[i ] +( 0 if i == length - 1 else channels[i]),
                    channels[i - 1],
                    1,
                    strides[i - 1],
                    dropout_upsample.pop() if attn[i-1] else None,
                    n_heads_upsample.pop() if attn[i - 1] else None,
                    resNetBlocks[i - 1],
                    upsample=True,
                    sample=True if i < length - 1 else False
                ) for i in range(length - 1, 0, -1)
            ]
                                               )

        self.end = ResNetBlock(
            1,
            2 * channels[0],
            inChannels
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:

        assert t.shape[0] == x.shape[0]

        time_emb = self.time_emb(t)
        x = torch.cat([x, time_emb], dim=1)
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


if __name__ == '__main__':

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    batch_size = 1
    image_shape = (218,178)
    sample_batch = torch.randn(batch_size,3,*image_shape, device=device, dtype=torch.float32)
    sample_batch.to(device)

    model = UNET(
        timeStep=1000,
        orginalSize=image_shape,
        inChannels=3,
        channels=[128,256,512],
        strides=[2,2],
        n_heads=[1],
        attn=[True, False, False],
        resNetBlocks=[2,2,2],
        dropout=[0.2]

    )
    model.to(device)
    print("params size", sum(p.numel() for p in model.parameters()))

    print(device)

    with torch.no_grad():
        output = model(sample_batch, torch.Tensor([5 for i in range(batch_size)]).to(device).int())
    print(output.shape)
