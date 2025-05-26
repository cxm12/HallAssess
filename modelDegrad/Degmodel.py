import faulthandler
faulthandler.enable()
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


def make_model(args):
    args.n_resblocks = 64
    args.n_feats = 256
    return DegModel(args=args)
    

class DegModel(nn.Module):
    def __init__(self, tsk=1, inchannel=1):
        super(DegModel, self).__init__()
        print('inchannel = ', inchannel)
        self.img_range = 1
        self.mean = torch.zeros(1, 1, 1, 1)
        self.task = tsk
        embed_dim = 32
        self.window_size = 8
        self.conv_first = nn.Conv2d(inchannel, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, 1, 3, 1, 1)
        
        self.down1 = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.res_block = ResidualBlock(256)
        self.res_block1 = ResidualBlock(256, 128)
        self.res_block2 = ResidualBlock(128, 64)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def forward(self, x, tsk=0):
        hr = x
        if tsk > 0:
            self.task = tsk
        # ~~~~~~~~~~~~ Head ~~~~~~~~~~~~~~~ #
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
                
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        res = self.res_block(d3, d3)

        u1 = self.up1(res)
        # print('u1.shape = ', u1.shape)  # [32, 128, 32, 32]
        u11 = self.res_block1(u1, torch.cat((u1, d2), dim=1))
        # print('u11.shape = ', u11.shape)  # [32, 128, 32, 32]
        u2 = self.up2(u11)
        # print('u2.shape = ', u2.shape)  # [32, 64, 64, 64]
        u21 = self.res_block2(u2, torch.cat((u2, d1), dim=1))
        # print('u21.shape = ', u21.shape)
        output = self.up3(u21)
        
        x = self.conv_last(output)
        # print('x.shape = ', x.shape, 'hr.shape = ', hr.shape)  # 
        # out = x
        if self.task == 1:
            out = hr - x
        else:
            out = x + hr
        return out / self.img_range + self.mean

    
class ResidualBlock(nn.Module):
    def __init__(self, channels, outchannels=0):
        super(ResidualBlock, self).__init__()
        if outchannels==0:
            outchannels = channels
            
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.conv_last = nn.Conv2d(channels,  outchannels, 3, 1, 1)

    def forward(self, x, cat):
        a = self.conv_last(self.block(cat))
        return x + a
