# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True)
        )
    def forward(self, input):
        x = self.conv(input)
        return x + input




class conv_1x1(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(conv_1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.conv(x)
        return x



# class res_block(nn.Module):
#
#     def __init__(self, in_ch, ratio):
#         super(res_block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, in_ch * ratio, 1, padding=1),
#             nn.BatchNorm2d(in_ch),
#             nn.ReLU6(inplace=True),
#             nn.Conv2d(in_ch * ratio, in_ch * ratio, 3, groups=in_ch * ratio, padding=1),
#             nn.BatchNorm2d(in_ch),
#             nn.ReLU6(inplace=True),
#             nn.Conv2d(in_ch * ratio, in_ch, 1, padding=1),
#             nn.BatchNorm2d(in_ch),
#             nn.ReLU6(inplace=True)
#         )

    # def forward(self, input):
    #     x = self.conv(input)
    #     return x




class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class down(nn.Module):
    def __init__(self, in_ch, out_ch, num_conv):
        super(down, self).__init__()
        self.conv = [nn.MaxPool2d(2),
                     conv_1x1(in_ch=in_ch, out_ch=out_ch)]
        for i in range(num_conv):
            self.conv.append(double_conv(in_ch=out_ch))
        self.conv = nn.Sequential(*self.conv)
    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, in_ch):
        super(ResidualConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        x = self.conv(input)
        return x



class MultiResolutionFusion(nn.Module):
    def __init__(self, low_ch, high_ch):
        super(MultiResolutionFusion, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(low_ch, high_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, low_x, high_x):
        low_x = self.conv(low_x)
        low_x = self.upsample(low_x)
        x = low_x + high_x
        return x


class ChainedPoolUnit(nn.Module):
    def __init__(self, in_ch):
        super(ChainedPoolUnit, self).__init__()
        self.conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False)
        )
    def forward(self, input):
        x = self.conv(input)
        return x


class ChainedResidualPooling(nn.Module):
    def __init__(self, in_ch):
        super(ChainedResidualPooling, self).__init__()
        self.modules = []
        self.chainPoolUnit_1 = ChainedPoolUnit(in_ch)

    def forward(self, input):
        x = nn.ReLU6(inplace=True)(input)
        path = x
        path = self.chainPoolUnit_1(path)
        x = x + path
        return x


class up(nn.Module):
    def __init__(self, low_ch, high_ch):
        super(up, self).__init__()
        self.residual_unit = ResidualConvUnit(in_ch=high_ch)
        self.fusion = MultiResolutionFusion(low_ch=low_ch, high_ch=high_ch)
    def forward(self, low_x, high_x):
        high_x = self.residual_unit(high_x)
        fusion_x = self.fusion(low_x, high_x)
        return fusion_x



class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(
             nn.Conv2d(in_ch, out_ch, 1),
             nn.Sigmoid()

        )

    def forward(self, x):
        x = self.conv(x)
        return x
