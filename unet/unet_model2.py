# full assembly of the sub-parts to form the complete net
#当前最好模型
from .unet_parts2 import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128, num_conv=1)
        self.down2 = down(128, 256, num_conv=1)
        self.down3 = down(256, 512, num_conv=1)
        self.down4 = down(512, 512, num_conv=1)
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        up1 = self.up1(x5, x4)
        up2 = self.up2(up1, x3)
        up3 = self.up3(up2, x2)
        up4 = self.up4(up3, x1)
        out1 = self.outc(up4)
        return out1
