""" Full assembly of the parts to form the complete network """

from unet.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up0 = (Up_fusion(2048, 512 // factor, bilinear))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outf = (OutConv(64, 256))
        self.outc = (OutConv(256, n_classes))

    def forward(self, x, y):
        x1, y1 = self.inc(x), self.inc(y)
        x2, y2 = self.down1(x1), self.down1(y1)
        x3, y3 = self.down2(x2), self.down2(y2)
        x4, y4 = self.down3(x3), self.down3(y3)
        x5, y5 = self.down4(x4), self.down4(y4)
        z = torch.cat([x5, y5], dim=1)
        z = self.up0(z, x4)
        x, y = self.up1(x5, x4), self.up1(y5, y4)
        z, x, y = self.up2(z, x3), self.up2(x, x3), self.up2(y, y3)
        z, x, y = self.up3(z, x2), self.up3(x, x2), self.up3(y, y2)
        z, x, y = self.up4(z, x1), self.up4(x, x1), self.up4(y, y1)
        z_fea, x_fea, y_fea = self.outf(z), self.outf(x), self.outf(y)
        logits = self.outc(z_fea)
        return logits, x_fea, y_fea

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=10)
    print(net)
