"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, skip_weight=1,prun = False):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        #self.skip_weight = nn.Parameter(torch.tensor(1.00))
        self.skip_weight = 1
        #self.skip_weight = skip_weight
        #shortcut
        self.shortcut = nn.Sequential()
        self.prun = prun
        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        #print(self.skip_weight)
        if self.prun:
            return nn.ReLU(inplace=True)(self.residual_function(x))
        else:
            return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )
        self.skip_weight = nn.Parameter(torch.tensor(0.5))
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        #print(self.skip_weight)
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, w, num_classes=100):
        super().__init__()

        self.in_channels = 64
        self.w = w
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, [1]*3)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, [1]*4)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, [1]*4)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, [1]*0)   #0是不剪去skip connection
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, pos):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        t = 0
        #prun_target = [3,4,5] #剪掉哪个skip connection
        if len(pos)>0:
            for stride in strides:
                if pos[t] == 1:
                #if p:
                    layers.append(block(self.in_channels, out_channels, stride, self.w[t], True))
                else:
                    layers.append(block(self.in_channels, out_channels, stride, self.w[t], False))
                    #print(out_channels)
                self.in_channels = out_channels * block.expansion
                t = t+1

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        #print(output.size())
        output = self.conv2_x(output)
        #print(output.size())
        output = self.conv3_x(output)
        #print(output.size())
        output = self.conv4_x(output)
        #print(output.size())
        output = self.conv5_x(output)
        #print(output.size())
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        #print(output.size())
        output = self.fc(output)

        return output

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    import numpy as np
    w = torch.ones(16)
    nn.init.normal_(w, mean=1, std=0.8)
    #print(w)
    cum_degree = np.cumsum(sorted(np.append(list(w), 0)))
    sum_degree = cum_degree[-1]
    xarray = np.array(range(0, len(cum_degree))) / np.float(len(cum_degree) - 1)
    yarray = cum_degree / sum_degree
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    G = A / (A + B)
    # print('G:',G)

    return ResNet(BasicBlock, [3,4,4,0], w)
    #return ResNet(BasicBlock, [3, 4, 6, 3])
def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])



