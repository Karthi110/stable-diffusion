import torch
from torch import nn
from torch.nn import functional as f
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # Reducing the size of the image.
            # (batch size,channel size,height,width)->(batch size,128,height,width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (batch size,128,height,with)->(batch size,128,height,width)
            VAE_ResidualBlock(128, 128),

            # (batch size,128,height,with)->(batch size,128,height,width)
            VAE_ResidualBlock(128, 128),

            # (batch size,128,height,width)->(batch size,128,height/2,width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (batch size,128,height/2,width/2)->(batch size,256,height/2,width/2)
            VAE_ResidualBlock(128, 256),

            # (batch size,256,height/2,width/2)->(batch size,256,height/2,width/2)
            VAE_ResidualBlock(256, 256),

            # (batch size,256,height/2,width/2)->(batch size,256,height/4,width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (batch size,256,height/4,width/4)->(batch size,512,height/4,width/4)
            VAE_ResidualBlock(256, 512),

            # (batch size,512,height/4,width/4)->(batch size,512,height/4,width/4)
            VAE_ResidualBlock(512, 512),

            # (batch size,512,height/4,width/4)->(batch size,512,height/8,width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (batch size,512,height/8,width/8)->(batch size,512,height/8,width/8)
            VAE_ResidualBlock(512, 512),

            # (batch size,512,height/8,width/8)->(batch size,512,height/8,width/8)
            VAE_ResidualBlock(512, 512),

            # (batch size,512,height/8,width/8)->(batch size,512,height/8,width/8)
            VAE_ResidualBlock(512, 512),

            # (batch size,512,height/8,width/8)->(batch size,512,height/8,width/8)
            VAE_AttentionBlock(512),

            # (batch size,512,height/8,width/8)->(batch size,512,height/8,width/8)
            VAE_ResidualBlock(512, 512),

            # (batch size,512,height/8,width/8)->(batch size,512,height/8,width/8)
            nn.GroupNorm(32, 512),

            # (batch size,512,height/8,width/8)->(batch size,512,height/8,width/8)
            nn.SiLU(),

            # (batch size,512,height/8,width/8)->(batch size,8,height/8,width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (batch size,8,height/8,width/8)->(batch size,8,height/8,width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x:(batch size,channel,height,width)
        # noise :(batch size,out channel,height/8,width/8)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (padding_left,padding_right,padding_top,padding_bottom)
                x = f.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (batch size,8,height/8,width/8)->two tensors of shape (batch size,4,height/8,width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # (batch size,4,height/8,width/8)->(batch size,4,height/8,width/8)
        log_variance = torch.clamp(log_variance, -30, 20)

        # (batch size,4,height/8,width/8)->(batch size,4,height/8,width/8)
        variance = log_variance.exp()

        # (batch size,4,height/8,width/8)->(batch size,4,height/8,width/8)
        stdev = variance.sqrt()

        # x=mean+stdev *z
        x = mean+stdev * noise

        # scale the output by aconstant
        x *= 0.18215

        return x
