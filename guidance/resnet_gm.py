import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import legendre
import numpy as np
from torch.autograd import Variable
from scipy.stats import ortho_group
import math

class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes):
    super().__init__()
    self.num_features = num_features
    self.bn = nn.InstanceNorm2d(num_features, affine=False)
    self.embed = nn.Linear(num_classes, num_features * 2, bias=False)

  def forward(self, x, y):
    out = x
    gamma, beta = self.embed(y).chunk(2, 1)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, k=3, p=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=k,padding=p,  stride=stride, bias=False)
        self.b = bn
        self.df = planes
        #if bn:
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=k,padding=p,
                              stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlockG(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, k=3, p=1):
        super(BasicBlockG, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=k, padding=p, stride=stride, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
 
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=k,padding=p, 
                               stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.shortcut = nn.Sequential()
       
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def moving_mean(x, training):
    if training:
        w = torch.cuda.FloatTensor(1, x.shape[1], 1, 1).normal_(1.0, 0.2)
        b = torch.cuda.FloatTensor(1, x.shape[1], 1, 1).normal_(0.0, 0.2)
        return x*w+b
    else:
        return x
    
def dropout_without_scale(x, p, training):
    if training:
        m = torch.ones_like(x)*p
        m = torch.bernoulli(m)
        return x*m
    else:
        return x
    
def brelu(x):
    x = F.leaky_relu(x)
    return 1. - F.leaky_relu((1.0 -x))

class BasicBlockGM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, n=1,  k=3, p=1, hw=32, box=None):
        super(BasicBlockGM, self).__init__()
        layers = []
        pl = in_planes
        for i in range(n):
            layers.append(BasicBlock(pl, planes, stride, bn, k, p)) 
            pl= planes      
        self.layer = nn.Sequential(*layers)

        self.conv1g = nn.Conv2d(
            in_planes, planes, kernel_size=1, padding=0, stride=1, bias=False)
                      
        self.bn1g = nn.BatchNorm2d(planes)
 
        self.conv2g = nn.Conv2d(planes, planes, kernel_size=1,padding=0, 
                                stride=1, bias=False)        
        self.bn2g = nn.BatchNorm2d(planes)

        self.fc1_1 = nn.Linear(in_planes, 2*planes)
        self.df = planes
        self.fc1_2 = nn.Linear(2*planes, 2*self.df, bias=True)
        self.fc1_3 = nn.Linear(2*planes, self.df, bias=True)
        self.fc1_4 = nn.Linear(2*planes, 2*2, bias=True)
        self.fc1_5 = nn.Linear(2*planes, 2, bias=True)
        self.ln11 = nn.LayerNorm(2*self.df)
        self.ln12 = nn.BatchNorm2d(self.df)
        self.ln13 = nn.BatchNorm2d(self.df)
        
        self.hw = hw
        self.box = box
        self.do1 = nn.Dropout(p=0.1)
        self.do2 = nn.Dropout(p=0.1)
        self.pw = nn.Parameter(Variable(torch.zeros(1, self.df, 1, 1), requires_grad=True))
        self.pw1 = nn.Parameter(Variable(torch.ones(1, self.df, 1, 1), requires_grad=True))
       
    def forward(self, x, xy1, grid, gridt, mask, ):
        out = self.layer(x +self.pw*grid)
        xy1_ = F.relu(self.ln11(self.fc1_1(xy1)))
        xy_b =  self.fc1_3(xy1_).view(-1, self.df, 1,1)
        xy_ = self.fc1_2(xy1_).view(-1,self.df , 2)
        xy_b1 =  self.fc1_5(xy1_).view(-1, 2, 1,1)
        xy_1 = self.fc1_4(xy1_).view(-1,2 , 2)

        gridt = torch.matmul(xy_1, gridt.view(-1, 2, self.hw*self.hw)).view(-1, 2, self.hw, self.hw) + xy_b1

        g1 = torch.matmul(xy_, gridt.view(-1, 2, self.hw*self.hw)).view(-1, self.df, self.hw, self.hw) + xy_b
        outg1 = F.relu(self.bn1g(self.conv1g(g1 +self.pw1*grid)))
        outg1 = F.relu(self.bn2g(self.conv2g(outg1)))

        grid =    outg1*out
        xy1 =  torch.flatten(F.avg_pool2d(grid, x.shape[2]), 1)
        return out, grid, xy1, outg1, gridt


class MyResNet1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, device=None):
        super(MyResNet1, self).__init__()

        self.device = device
        self.hw8 = 8
        self.hw = 32
        self.df=256
        h = (self.hw-1)
        a = (torch.Tensor(range(self.hw)))/(h)
        g = torch.meshgrid(a, a)
        self.gridt = nn.Parameter(torch.cat((g[0].view(1, 1, self.hw,self.hw), g[1].view(1, 1, self.hw,self.hw),
                        ),dim=1), requires_grad=False)
        self.grid = nn.Parameter(self.gridt.view(-1, 2, self.hw*self.hw), requires_grad=False)
        self.mask = nn.Parameter(torch.ones(1,1,self.hw,self.hw), requires_grad=False)
        self.xy_ = nn.Parameter(Variable(torch.rand(1, self.df, 2), requires_grad=True))
        self.xy_b = nn.Parameter(Variable(torch.rand(1,self.df, 1,1), requires_grad=True))
        self.box = nn.Parameter(torch.Tensor([[-1.0, -1.0], [1.0, 1.0]]).float(), requires_grad=False)

        self.in_planes = self.df
        self.in_planes = self.df
        self.layer01 = self._make_layer(BasicBlockG, self.df, 1, stride=1, k=1, p=0)
        self.layer02 = self._make_layer(block, self.df, 4, stride=1, k=3, p=1)
        self.in_planes = self.df
        self.conv02 = nn.Conv2d(3, self.df, kernel_size=8, stride=8, padding=0)
        self.in02 = nn.BatchNorm2d(self.df)
        self.resgm1 = BasicBlockGM(self.df, self.df, n=4, k=3, p=1, hw=self.hw, box = self.box)
        self.resgm2 = BasicBlockGM(self.df, self.df, n=4, k=3, p=1, hw=self.hw, box = self.box)
        self.resgm3 = BasicBlockGM(self.df, self.df, n=4, k=3, p=1, hw=self.hw, box = self.box)
        self.resgm4 = BasicBlockGM(self.df, self.df, n=2, k=3, p=1, hw=self.hw, box = self.box)

        self.sf = nn.Softmax2d()
        self.linear = nn.Linear(self.df, num_classes)

        dropout = 0.1
        self.do1 = nn.Dropout(p=dropout)
        self.conv11 = nn.Conv2d(2, self.df, kernel_size=1, stride=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride, k=3, p=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, k=k, p=p))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        im = x
        size = (x.shape[2], x.shape[3])
        gridt = self.gridt
        grid11 = self.layer01(self.conv11(gridt))
        grid1 = grid11
        x = self.conv02(im)
        x = self.layer02(x)
        grid = grid11*x
        xy1 = torch.flatten(F.avg_pool2d(grid, x.shape[2]), 1)
        x, grid, xy1, box, gridt = self.resgm1(x,xy1,grid, gridt, grid11)
        x, grid, xy1, box, gridt = self.resgm2(x,xy1,grid, gridt, box)
        x, grid, xy1, box, gridt = self.resgm3(x,xy1,grid, gridt, box)
        feat = xy1
        grid1 = grid
        x, grid, xy1, box, gridt = self.resgm4(x,xy1,grid, gridt, box)

        imgr = torch.sum(grid1*(feat).view(-1, xy1.shape[1], 1, 1), dim=1, keepdim=True)

        imgr = imgr.view(imgr.size(0), -1)
        imgr = imgr - imgr.min(1, keepdim=True)[0]
        imgr = imgr/imgr.max(1, keepdim=True)[0]
        imgr = (imgr.view(-1, 1, self.hw, self.hw))
        imgr = nn.Upsample((512,512), mode='bilinear', align_corners=True)(imgr)
        
        cl=self.linear(xy1)
        return cl, imgr


def ResNet18(device=None, num_classes=1000):
    return MyResNet1(BasicBlock, [1, 1, 1, 1], device=device, num_classes=num_classes)


def ResNet34(device=None, num_classes=1000):
    return MyResNet1(BasicBlock, [3, 4, 6, 3], device=device, num_classes=num_classes)
