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


class Attention(nn.Module): 

    def __init__(self, channels) 



    def forward() 




class DownSample(nn.Module): 

    def __init__(self, ):
        
        self.block = nn.Sequential(
            self.layer1 = ResNetBlock() , 
            self.attn = Attention() , 
            self.layer2 = ResNetBlock() , 
            ) 



    def forward(self, x): 
        
        return self.block(x)

class UpSample(nn.Module): 

    def __init__(self): 

        self.block = nn.Sequential( 
                self.layer1 = ResNetBlock() , 
                self.attn = Attention(), 
                self.layer2 = ResnetBlock(), 
                                   ) 

    def forwad(self, x): 
        return self.block(x) 
                        


class UNET(nn.Module): 

    def __init__(self): 

        super().__init__() 
        


    

    def forward()


