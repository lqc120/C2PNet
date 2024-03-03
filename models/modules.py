import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class SA_Module(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(SA_Module, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv0 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=channel),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1), padding=0),
        )

    def forward(self, x):
        x = self.conv0(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(cat)
        return self.sigmoid(out)

class ECA_Module(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA_Module, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return x * v

# Inverted Residual
class IR_Block(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(IR_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes * 2, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(out_planes * 2),
            nn.Conv2d(out_planes * 2, out_planes * 2, kernel_size=(3, 3), padding=(1, 1), groups=out_planes * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes * 2, out_planes, kernel_size=(1, 1), stride=(1, 1), padding=0),
        )

    def forward(self, input):
        return self.conv(input) + input

class Multi_CIF(nn.Module):
    def __init__(self, c0, c1, c2, c3):
        super(Multi_CIF, self).__init__()
        self.sa0 = SA_Module(c0)
        self.sa1 = SA_Module(c1)
        self.sa2 = SA_Module(c2)
        self.sa3 = SA_Module(c3)
        self.avg_pool0 = nn.AvgPool2d(2, stride=2)
        self.avg_pool1 = nn.AvgPool2d(2, stride=2)
        self.avg_pool2 = nn.AvgPool2d(2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv01 = nn.Conv2d(c0, c1, kernel_size=(1, 1), stride=(1, 1))
        self.conv10 = nn.Conv2d(c1, c0, kernel_size=(1, 1), stride=(1, 1))
        self.conv12 = nn.Conv2d(c1, c2, kernel_size=(1, 1), stride=(1, 1))
        self.conv21 = nn.Conv2d(c2, c1, kernel_size=(1, 1), stride=(1, 1))
        self.conv23 = nn.Conv2d(c2, c3, kernel_size=(1, 1), stride=(1, 1))
        self.conv32 = nn.Conv2d(c3, c2, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x0, x1, x2, x3):
        s0 = self.sa0(x0)
        s1 = self.sa1(x1)
        s2 = self.sa2(x2)
        s3 = self.sa3(x3)
        y0 = self.conv10(self.up(x1)) * s0 + s0 * x0 + x0
        y1 = self.conv01(self.avg_pool0(x0)) * s1 + self.conv21(self.up(x2)) * s1 + s1 * x1 + x1
        y2 = self.conv12(self.avg_pool1(x1)) * s2 + self.conv32(self.up(x3)) * s2 + s2 * x2 + x2
        y3 = self.conv23(self.avg_pool2(x2)) * s3 + s3 * x3 + x3
        return y0, y1, y2, y3

class CGPC(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(CGPC, self).__init__()
        self.channel = out_planes//4
        self.conv00 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(out_planes),
        )
        self.conv1 = nn.Conv2d(self.channel, self.channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                      dilation=(1, 1))
        self.conv2 = nn.Conv2d(self.channel, self.channel, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2),
                      dilation=(2, 2))

        self.conv3 = nn.Conv2d(self.channel, self.channel, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4),
                      dilation=(4, 4))
        self.irac1 = nn.Conv2d(self.channel, self.channel, kernel_size=(3, 3), padding=(1, 1), groups=self.channel)
        self.irac2 = nn.Conv2d(self.channel, self.channel, kernel_size=(5, 5), padding=(2, 2), groups=self.channel)
        self.irac3 = nn.Conv2d(self.channel, self.channel, kernel_size=(7, 7), padding=(3, 3), groups=self.channel)
        self.conv5 = nn.Conv2d(out_planes, out_planes, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv6 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=(1, 1), stride=(1, 1), padding=0),
        )
        self.conv7 = IR_Block(out_planes, out_planes)
        self.eca = ECA_Module(out_planes)


    def forward(self, x):
        r = self.conv00(x)
        x = self.eca(r)
        x0, x1, x2, x3 = torch.chunk(x, 4, dim=1)
        x1 = self.irac1(x1)
        x11 = self.conv1(x1+x0)
        x2 = self.irac2(x2)
        x22 = self.conv2(x2+x1)
        x3 = self.irac3(x3)
        x33 = self.conv3(x3+x2)
        x4 = torch.cat((x0, x11, x22, x33), dim=1)
        y1 = self.conv5(x4) + r
        y2 = self.conv6(y1) + y1
        y3 = self.conv7(y2)
        return y3






