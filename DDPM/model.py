import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from typing import List, Set, Dict, Tuple, Optional


class ResNetBlock(nn.Module):

    def __init__(self, kernels: Tuple[Tuple[int,int], Tuple[int,int]],
                 channels: Tuple[int,int,int],
                 num_groups: Tuple[int,int],
                 stride: Tuple[int,int],
                 padding: Tuple[int,int],
                 device: str,
                 upsample: Optional[bool] =False ) -> None:

        super().__init()

        arg1 = (channels[0], channels[1], kernels[0], stride[0], padding[0], device)
        arg2 = (channels[1], channels[2], kernels[1], stride[1], padding[1], device)

        self.network = nn.Sequential(
                        nn.ConvTranspose2d(*arg1) if upsample else nn.Conv2d(*arg2),
                        nn.GroupNorm(num_groups),
                        nn.Relu(),
                        nn.ConvTransposed2d(*arg1) if upsample else nn.Conv2d(*arg2),
                        nn.GroupNorm(num_groups)
                        )

        self.relu = nn.Relu()


    def forward(self, x: Tensor[Tensor[Tensor[Tensor[float]]]]) -> Tensor[Tensor[Tensor[Tensor[float]]]]:
        x = self.network(x) + x

        logits = self.relu(x)
        return logits

class TimeEmbedding(nn.Module):

    def __init__(self, device):


        #TODO make positional embeeding
        self.pos_embedding

    def forward(self, x):


        t_emb = self.pos_embedding(x)

        return t_emb

class Attention(nn.Module):

    def __init__(self,
                 n_heads: int,
                 in_channel: int,
                 n_emd: int,
                 dropout: float,
                 device: Optional[str]) -> None:

        super().__init__()
        assert n_emd % n_heads == 0
        self.qkv = nn.Conv2d(in_channel, 3 * in_channel,kernel_size=1, stride=1, padding=0) # combine qkv for efficency
        self.proj = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.emd = n_emd // n_heads # embedding per head

    def forward(self, x: Tensor[Tensor[Tensor[Tensor[float]]]]) -> Tensor[Tensor[Tensor[Tensor[float]]]]:

        b,c,h,w = x.shape
        # pass through projection
        qkv = self.qkv(x)
        qkv = qkv.view(b,3 * self.n_heads, c // self.n_heads, h * w)
        qkv = qkv.transpose(-2, -1) # switch channels with time steps ( height and width)
        q,k,v = qkv.spilt(self.n_heads, dim=1)
        k_T = k.transpose(2,3)
        attn = F.softmax(((q @ k_T)/torch.sqrt(self.emb) ), dim=-1)
        self.logits = attn @ v
        self.logits = self.dropout(self.logits)
        self.logits = torch.cat(self.logits, dim=1) # cat heads
        self.logits = self.logits.permute(0,2,1) # reshape back into images
        self.logits = self.logits.view(b, c, h,w )
        return self.proj(self.logits)

class Sample(nn.Module):

    def __init__(self,
                 kernels: Tuple[Tuple[Tuple[int,int], Tuple[int,int]], Tuple[Tuple[int,int], Tuple[int,int]]],
                 channels: Tuple[Tuple[int,int,int], Tuple[int,int,int]],
                 num_groups: Tuple[Tuple[int,int], Tuple[int,int]],
                 stride: Tuple[Tuple[int,int], Tuple[int,int]],
                 padding: Tuple[Tuple[int, int], Tuple[int,int]],
                 n_heads: int,
                 n_emb: int,
                 dropout: float,
                 sample: Optional[str],
                 device: Optional[str]) -> None:

        super().__init__( )


        assert channels[0][-1] == channels[1][0] # check for continunity

        in_channel = channels[0][2]
        ResNetBlock_1_config = (kernels[0], channels[0], num_groups[0], stride[0], padding[0], device)
        attn_config = (n_heads, in_channel, n_emb, dropout, device)
        ResNetBlock_2_config = (kernels[1], channels[1], num_groups[1], stride[1], padding[1], device)

        self.block = nn.Sequential(
                ResNetBlock(*ResNetBlock_1_config, upsample=sample) ,
                Attention(*attn_config),
                ResNetBlock(*ResNetBlock_2_config, upsample=sample),
        )

    def forward(self, x):
        return self.block(x)

class UNET(nn.Module):

    def __init__(self,
                 kernels: List[Tuple[Tuple[Tuple[int,int], Tuple[int,int]], Tuple[Tuple[int,int], Tuple[int,int]]]],
                 channels: List[Tuple[Tuple[int,int,int], Tuple[int,int, int]]],
                 num_groups: List[Tuple[Tuple[int,int], Tuple[int,int]]],
                 strides: List[Tuple[Tuple[int,int], Tuple[int,int]]],
                 paddings: List[Tuple[Tuple[int,int], Tuple[int,int]]],
                 n_heads: List[int],
                 n_emb: List[int],
                 dropout: List[float],
                 device: Optional[str]
                 ) -> None :

        super().__init__()

        # checks

        # assert last channels of one layer is the input of the next
        for i, ch in enumerate(channels):
            if (i + 1 != len(channels)):
                assert ch[-1][-1] == (channels[i+1][0][0])
            else:
                assert ch[0][0] == ch[-1][-1]

        self.downsample_layers = [
                                    Sample(kernels[i],
                                           channels[i],
                                           num_groups[i],
                                           strides[i],
                                           paddings[i],
                                           n_heads[i],
                                           n_emb[i],
                                           dropout[i],
                                           device,
                                           sample=False)
                                    for i in range(len(kernels))
                                  ]

        upsample_channels = []
        for i in range(1, len(channels)) :
            a,b,c = channels[i][0]
            d,e,f = channels[i][1]
            upsample_channels.append(((2 * f, e, d), (c,b,a)))


        self.upsample_layers = [Sample(kernels[i],
                                           upsample_channels[i],
                                           num_groups[i],
                                           strides[i],
                                           paddings[i],
                                           n_heads[i],
                                           n_emb[i],
                                           dropout[i],
                                           device,
                                           sample=True)
            for i in range(len(upsample_channels) - 1, -1, -1)
        ]

    def forward(self, x: Tensor[Tensor[Tensor[Tensor[float]]]]) -> Tensor[Tensor[Tensor[Tensor[float]]]]:

        downSample = []

        # downsample
        for block in self.downsample:
            x = block(x)
            downSample.append(x)

        downSample.pop() # we don't need last element since that is a skip connection and not residual

        # upsample
        for block in self.upsample_layers:
            skip_connection = downSample.pop()
            x = torch.cat(x, skip_connection, dim=1)
            x = block(x)

        return x


if __name__ == '__main__':
