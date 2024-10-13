import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class ResNetBlock(nn.Module): 

    def __init__(self, kernels, channels, num_groups,  stride, padding, upsample=False ): 

        super().__init() 

        arg1 = (channels[0], channels[1], kernels[0], stride[0], padding[0]) 
        arg2 = (channels[1], channeps[2], kernels[1], stride[1], padding[1])

        self.network = nn.Sequential(
                        nn.ConvTranspose2d(*arg1) if upsample else nn.Conv2d(*arg2), 
                        nn.GroupNorm(num_groups), 
                        nn.Relu(), 
                        nn.ConvTransposed2d(*arg1) if upsample else nn.Conv2d(*arg2), 
                        nn.GroupNorm(num_groups)
                        ) 
        
        self.relu = nn.Relu() 


    def forward(self, x):
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

    def __init__(self, head_size, in_channel, n_emd, dropout, device): 
        
        super().__init__() 

        assert n_emd % head_size == 0

        self.qkv = nn.Conv2d(in_channel, 3 * in_channel,kernel_size=1, stride=1, padding=0) 
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size 
        self.emd = n_emd


    
    def forwad(self, x): 

        B,H,W,C = x.shape 
        
        qkv = self.qkv(x) 
        qkv = qkv.view(B, C, H*W) 
        qkv = qkv.permute(0, 2, 1) 

        q,k,v = qkv.spilt(self.emb, dim=1) 

        k = k.permute(0, 1, 3,2)  # transpose 
        
        
        attn = F.softmax(((q @ k)/torch.sqrt(self.emb) ), dim=-1) 

        self.logits = attn @ v 
        
        self.logits = self.dropout(self.logits) 

        return self.logits 



class DownSample(nn.Module): 

    def __init__(self, device):
        
        self.block = nn.Sequential(
            self.layer1 = ResNetBlock() , 
            self.attn = Attention() , 
            self.layer2 = ResNetBlock() , 
            ) 



    def forward(self, x): 
        
        return self.block(x)

class UpSample(nn.Module): 

    def __init__(self,device): 

        self.block = nn.Sequential( 
                self.layer1 = ResNetBlock() , 
                self.attn = Attention(), 
                self.layer2 = ResnetBlock(), 
                                   ) 

    def forwad(self, x): 
        return self.block(x) 
                        


class UNET(nn.Module): 

    def __init__(self, device): 

        super().__init__() 
        self.queue = torch.zeros( ,device=device) 


    

    def forward()


