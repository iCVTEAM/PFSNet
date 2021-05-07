

#!/usr/bin/python3
#coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU,nn.AdaptiveAvgPool2d,nn.Softmax,nn.Dropout2d)):
            pass
        else:
            m.initialize()

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out2 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out2)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1,out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('/media/pcl/Bdata/MMC/D3Net/PFSNet/resnet/resnet50-19c8e357.pth'), strict=False)

class Rblock(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(Rblock,self).__init__()
        self.squeeze1 = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=3,stride=1,dilation=2,padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=3,stride=1,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.convg = nn.Conv2d(128,128,1)
        self.sftmax = nn.Softmax(dim=1)    
        self.convAB = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bnAB   = nn.BatchNorm2d(64)

    def forward(self, x,z):
        z = self.squeeze1(x) if z is None else self.squeeze1(x+z)
        x = self.squeeze2(x)
        x = torch.cat((x,z),1)
        y = self.convg(self.gap(x))       
        x = torch.mul(self.sftmax(y)*y.shape[1],x)     
        x = F.relu(self.bnAB(self.convAB(x)),inplace=True)        
        return x,z
    def initialize(self):
        weight_init(self)
        print("Rblock module init")

class Yblock(nn.Module):
    def __init__(self):
        super(Yblock,self).__init__()

        self.convA1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bnA1   = nn.BatchNorm2d(64)
        
        self.convB1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bnB1   = nn.BatchNorm2d(64)
        
        self.convAB = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bnAB   = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.convg = nn.Conv2d(128,128,1)
        self.sftmax = nn.Softmax(dim=1)
      #  self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x,y):
        if x.size()[2:] != y.size()[2:]:
            y = F.interpolate(y, size=x.size()[2:], mode='bilinear',align_corners=True)
        fuze = torch.mul(x,y)
        y = F.relu(self.bnB1(self.convB1(fuze+y)),inplace=True)
        x = F.relu(self.bnA1(self.convA1(fuze+x)),inplace=True)
        x = torch.cat((x,y),1)
        y = self.convg(self.gap(x))
        x = torch.mul(self.sftmax(y)*y.shape[1],x)     
     #   x = self.dropout(x)
        x = F.relu(self.bnAB(self.convAB(x)),inplace=True)
        return x

    def initialize(self):
        weight_init(self)
        print("Yblock module init")

class net(nn.Module):
    def __init__(self, cfg=None):
        super(net, self).__init__()
        self.cfg      = cfg
        self.bkbone   = ResNet()
        self.squeeze5 = Rblock(2048,64)
        self.squeeze4 = Rblock(1024,64)
        self.squeeze3 = Rblock(512,64)
        self.squeeze2 = Rblock(256,64)
        self.squeeze1 = Rblock(64,64)

        self.conv1 = nn.Conv2d(64, 1024, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(1024)

        self.conv2 = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(512)

        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(64)

        self.Y11=Yblock()
        self.Y12=Yblock()
        self.Y13=Yblock()
        self.Y14=Yblock()

        self.Y21=Yblock()
        self.Y22=Yblock()
        self.Y23=Yblock()

        self.Y31=Yblock()
        self.Y32=Yblock()

        self.Y41=Yblock()

        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.initialize()

    def forward(self, x, shape=None):

        s1,s2,s3,s4,s5 = self.bkbone(x)

        s5,z=self.squeeze5(s5,None)
        z= F.interpolate(z, size=s4.size()[2:],mode='bilinear',align_corners=True)
        z=F.relu(self.bn1(self.conv1(z)))
        s4,z=self.squeeze4(s4,z)
        z= F.interpolate(z, size=s3.size()[2:],mode='bilinear',align_corners=True)
        z=F.relu(self.bn2(self.conv2(z)))
        s3,z=self.squeeze3(s3,z)
        z= F.interpolate(z, size=s2.size()[2:],mode='bilinear',align_corners=True)
        z=F.relu(self.bn3(self.conv3(z)))
        s2,z=self.squeeze2(s2,z)
        z= F.interpolate(z, size=s1.size()[2:],mode='bilinear',align_corners=True)
        z=F.relu(self.bn4(self.conv4(z)))
        s1,z=self.squeeze1(s1,z)

        s5 = self.Y14(s4,s5)
        s4 = self.Y13(s3,s4)
        s3 = self.Y12(s2,s3)
        s2 = self.Y11(s1,s2)

        s2 = self.Y21(s2,s3)
        s3 = self.Y22(s3,s4)
        s4 = self.Y23(s4,s5)

        s4 = self.Y32(s3,s4)
        s3 = self.Y31(s2,s3)

        s3 = self.Y41(s3,s4)

        shape = x.size()[2:] if shape is None else shape
        p1 = F.interpolate(self.linearp1(s3), size=shape, mode='bilinear',align_corners=True)
        del s1,s2,s3,s4,s5,z          
        torch.cuda.empty_cache()
        return p1


    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)

