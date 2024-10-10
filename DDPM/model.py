import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class ResNetBlock(nn.Module): 

    def __init__(self, channels,upsample=False ): 

        super().__init() 
        self.network = nn.Sequential(
                        nn.ConvTranspose2d if upsample else nn.Conv2d(), 
                        nn.GroupNorm(), 
                        nn.Relu(), 
                        nn.ConvTransposed2d if upsample else nn.Conv2d(), 
                        nn.GroupNorm()
                        ) 
        
        self.relu = nn.Relu() 


    def forward(self, x):
        x = self.network(x) + x

        logits = self.relu(x)
        return logits 


class Attention(nn.Module): 

    def __init__(self, channels) 



    def forward() 




class DownSample(nn.Module)

    def __init__(self) 
        
        self.block = nn.Sequential(
            self.layer1 = ResNetBlock() 
            self.attn = Attention() 
            self.layer2 = ResNetBlock() 
            ) 



    def forward(self, x): 
        
        x = self.layer1(x) 
        x = self.attn


class UpSample(nn.Module)


class UNET(nn.Module): 

    def __init__(self): 

        super().__init__() 


    

    def forward()


