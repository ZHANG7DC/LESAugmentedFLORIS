import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn

class UNetp(nn.Module):
    def __init__(self, in_channels, out_im_channels, mlp_in_channels, mlp_out_channels, batchnorm=True, dropout=0.0, bc=64):
        super(UNetp, self).__init__()

        self.inc = inconv(in_channels, bc*1, batchnorm)
        self.down1 = down(bc*1, bc*2, batchnorm, dropout=dropout)
        self.down2 = down(bc*2, bc*4, batchnorm, dropout=dropout)
        self.down3 = down(bc*4, bc*8, batchnorm, dropout=dropout)
        self.down4 = down(bc*8, bc*8, batchnorm, dropout=dropout)
        self.conv = nn.Conv2d(bc*8+mlp_out_channels, bc*8, 1)
        self.up1 = up(bc*16, bc*4, batchnorm, dropout=dropout)
        self.up2 = up(bc*8, bc*2, batchnorm, dropout=dropout)
        self.up3 = up(bc*4, bc*1, batchnorm, dropout=dropout)
        self.up4 = up(bc*2, bc*2, batchnorm, dropout=dropout)
        self.outc = outconv(bc*2, out_im_channels)
        self.mlp = TransEncoder(mlp_in_channels, mlp_out_channels, dropout)
    def forward(self, x, c):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = x5 = self.down4(x4)
        c = self.mlp(c).unsqueeze(-1).unsqueeze(-1).repeat(1,1, x5.shape[-2],x5.shape[-1])
        x6 = torch.cat([x,c],dim=1)
        x = self.conv(x6) + x
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
# --- helper modules --- #
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
      nn.ReLU(inplace=True),
    )

class TransEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.transencoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=out_ch, nhead=4, dim_feedforward=32, dropout=dropout), num_layers=4)
        self.mlp = nn.Sequential(
        nn.Linear(in_ch,out_ch),
        nn.ReLU(inplace=True),
        nn.Linear(out_ch,out_ch),
        nn.ReLU(inplace=True),
        nn.Linear(out_ch,out_ch))
    def forward(self, x):
        param_shape = x.shape
        return self.transencoder(self.mlp(x.view(-1,param_shape[-1]).float()).view(param_shape[0],param_shape[1],-1).permute(1,0,2))[0]

    
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, batchnorm=True):
        super(double_conv, self).__init__()
        if batchnorm:
            self.conv = nn.Sequential(
              nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
              nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(
              nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, batchnorm):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, batchnorm)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, batchnorm, dropout=None):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch, batchnorm))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    def forward(self, x):
        x = self.mpconv(x)

        if self.dropout:
            x = self.dropout(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, batchnorm, method='conv', dropout=None):
        super(up, self).__init__()

        if method == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif method == 'conv':
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        elif method == 'upconv':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReflectionPad2d(1),
                # note the interesting size and stride
                nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2, padding=0),
            )
        elif method == 'none':
            self.up = nn.Identity()

        self.conv = double_conv(in_ch, out_ch, batchnorm)

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # up conv here

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        if self.dropout:
            x = self.dropout(x)

        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class TDUNet(nn.Module):
    def __init__(self, in_channels, out_im_channels, batchnorm=True, dropout=0.0, bc=1):
        super(TDUNet, self).__init__()

        self.inc = tdinconv(in_channels, bc*1, batchnorm)
        self.down1 = tddown(bc*1, bc*2, batchnorm, dropout=dropout)
        self.down2 = tddown(bc*2, bc*4, batchnorm, dropout=dropout)
        self.down3 = tddown(bc*4, bc*8, batchnorm, dropout=dropout)
        self.down4 = tddown(bc*8, bc*8, batchnorm, dropout=dropout)
        self.up1 = tdup(bc*16, bc*4, batchnorm, dropout=dropout)
        self.up2 = tdup(bc*8, bc*2, batchnorm, dropout=dropout)
        self.up3 = tdup(bc*4, bc*1, batchnorm, dropout=dropout)
        self.up4 = tdup(bc*2, bc*2, batchnorm, dropout=dropout)
        self.outc = tdoutconv(bc*2, out_im_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)
# --- helper modules --- #
def tdconvrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
      nn.Conv3d(in_channels, out_channels, kernel, padding=padding),
      nn.ReLU(inplace=True),
    )


class tddouble_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, batchnorm=True):
        super(tddouble_conv, self).__init__()
        if batchnorm:
            self.conv = nn.Sequential(
              nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
              nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(
              nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
              nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class tdinconv(nn.Module):
    def __init__(self, in_ch, out_ch, batchnorm):
        super(tdinconv, self).__init__()
        self.conv = tddouble_conv(in_ch, out_ch, batchnorm)

    def forward(self, x):
        x = self.conv(x)
        return x


class tddown(nn.Module):
    def __init__(self, in_ch, out_ch, batchnorm, dropout=None):
        super(tddown, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool3d(2), tddouble_conv(in_ch, out_ch, batchnorm))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    def forward(self, x):
        x = self.mpconv(x)

        if self.dropout:
            x = self.dropout(x)
        return x


class tdup(nn.Module):
    def __init__(self, in_ch, out_ch, batchnorm, method='conv', dropout=None):
        super(tdup, self).__init__()

        self.up = nn.ConvTranspose3d(in_ch // 2, in_ch // 2, 2, stride=2)
        

        self.conv = tddouble_conv(in_ch, out_ch, batchnorm)

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # up conv here

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, (diffZ // 2, diffZ - diffZ // 2, diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        if self.dropout:
            x = self.dropout(x)

        return x


class tdoutconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(tdoutconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x