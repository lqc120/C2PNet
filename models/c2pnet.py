import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.modules import Multi_CIF, CGPC
from models.p2t import p2t_tiny, p2t_small


class C2PNet(nn.Module):
    def __init__(self, arch='p2t_tiny', pretrained=True):
        super(C2PNet, self).__init__()
        self.arch = arch
        self.backbone = eval(arch)(pretrained)
        dec_channels = [48, 96, 240, 384]
        self.cif = Multi_CIF(dec_channels[0], dec_channels[1], dec_channels[2], dec_channels[3])
        self.cgpc0 = CGPC(dec_channels[0]+dec_channels[1], dec_channels[0])
        self.cgpc1 = CGPC(dec_channels[1]+dec_channels[2], dec_channels[1])
        self.cgpc2 = CGPC(dec_channels[2]+dec_channels[3], dec_channels[2])
        self.cgpc3 = CGPC(dec_channels[3], dec_channels[3])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out = nn.Conv2d(dec_channels[0], 1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, input):
        size = input.size()
        backbone = self.backbone(input)
        q0, q1, q2, q3 = self.cif(backbone[0], backbone[1], backbone[2], backbone[3])
        f3 = self.cgpc3(q3)
        f2 = self.cgpc2(torch.cat((q2, self.up(f3)), dim=1))
        f1 = self.cgpc1(torch.cat((q1, self.up(f2)), dim=1))
        ff0 = self.cgpc0(torch.cat((q0, self.up(f1)), dim=1))
        f0 = self.out(ff0)
        f00 = F.interpolate(f0, size=(size[2], size[3]), mode='bilinear')
        return f00


if __name__ == '__main__':
    x = torch.randn(1, 3, 352, 352).cuda()
    slicing = C2PNet().cuda(0)
    y = slicing(x)
    print('y.shape', y.shape)
