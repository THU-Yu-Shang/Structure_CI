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
    def __init__(self, in_channels, out_channels, stride=1, skip_weight=1, prun=False):
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
        #self.skip_weight = nn.Parameter(torch.tensor(0.5))
        self.shortcut = nn.Sequential()
        self.prun = prun
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        if self.prun:
            return nn.ReLU(inplace=True)(self.residual_function(x))
        else:
            return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, w, num_classes=100):
        super().__init__()

        self.in_channels = 64
        self.w = w
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #self.probe1 = nn.Linear(64, num_classes)
        self.conv2_0 = self._make_layer(block, 64,  1, [1])
        #self.probe2 = nn.Linear(64, num_classes)
        self.conv2_1 = self._make_layer(block, 64,  1, [1])
        #self.probe3 = nn.Linear(64, num_classes)
        self.conv2_2 = self._make_layer(block, 64,  1, [1])
        #self.probe4 = nn.Linear(64, num_classes)
        self.conv3_0 = self._make_layer(block, 128,  2, [1])
        #self.probe5 = nn.Linear(128, num_classes)
        self.conv3_1 = self._make_layer(block, 128,  1, [1])
        #self.probe6 = nn.Linear(128, num_classes)
        self.conv3_2 = self._make_layer(block, 128,  1, [1])
        #self.probe7 = nn.Linear(128, num_classes)
        self.conv3_3 = self._make_layer(block, 128,  1, [1])
        #self.probe8 = nn.Linear(128, num_classes)
        self.conv4_0 = self._make_layer(block, 256,  2, [1])
        #self.probe9 = nn.Linear(256, num_classes)
        self.conv4_1 = self._make_layer(block, 256,  1, [1])
        #self.probe10 = nn.Linear(256, num_classes)
        self.conv4_2 = self._make_layer(block, 256,  1, [1])
        #self.probe11 = nn.Linear(256, num_classes)
        self.conv4_3 = self._make_layer(block, 256,  1, [1])
        #self.probe12 = nn.Linear(256, num_classes)
        self.conv4_4 = self._make_layer(block, 256,  1, [1])
        #self.probe13 = nn.Linear(256, num_classes)
        self.conv4_5 = self._make_layer(block, 256,  1, [1])
        #self.probe14 = nn.Linear(256, num_classes)
        self.conv5_0 = self._make_layer(block, 512,  2, [1])   #0是不剪去skip connection
        #self.probe15 = nn.Linear(512, num_classes)
        self.conv5_1 = self._make_layer(block, 512,  1, [1])
        #self.probe16 = nn.Linear(512, num_classes
        self.conv5_2 = self._make_layer(block, 512,  1, [1])
        #self.probe17 = nn.Linear(512, num_classes)
        #self.conv5_3 = self._make_layer(block, 512,  1, [1])
        #self.probe18 = nn.Linear(512, num_classes)
        #self.conv5_4 = self._make_layer(block, 512,  1, [1])
        #self.probe19 = nn.Linear(512, num_classes)
        #self.conv5_5 = self._make_layer(block, 512,  1, [1])
        #self.probe20 = nn.Linear(512, num_classes)
        #self.conv5_6 = self._make_layer(block, 512,  1, [1])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        

    def _make_layer(self, block, out_channels, stride, pos):
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
        layers = []
        #prun_target = [3,4,5] #剪掉哪个skip connection

        if pos[0] == 1:
            layers.append(block(self.in_channels, out_channels, stride, 1, True))
        else:
            layers.append(block(self.in_channels, out_channels, stride, 1, False))
            #print(out_channels)
        self.in_channels = out_channels * block.expansion


        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output1 = self.avg_pool(output)
        output1 = output1.view(output1.size(0), -1)
        #output1 = self.probe1(output1)

        output = self.conv2_0(output)
        output2 = self.avg_pool(output)
        output2 = output2.view(output2.size(0), -1)
        #output2 = self.probe2(output2)

        output = self.conv2_1(output)
        output3 = self.avg_pool(output)
        output3 = output3.view(output3.size(0), -1)
        #output3 = self.probe3(output3)

        output = self.conv2_2(output)
        output4 = self.avg_pool(output)
        output4 = output4.view(output4.size(0), -1)
        #output4 = self.probe4(output4)

        output = self.conv3_0(output)
        output5 = self.avg_pool(output)
        output5 = output5.view(output5.size(0), -1)
        #output5 = self.probe5(output5)

        output = self.conv3_1(output)
        output6 = self.avg_pool(output)
        output6 = output6.view(output6.size(0), -1)
        #output6 = self.probe6(output6)

        output = self.conv3_2(output)
        output7 = self.avg_pool(output)
        output7 = output7.view(output7.size(0), -1)
        #output7 = self.probe7(output7)

        output = self.conv3_3(output)
        output8 = self.avg_pool(output)
        output8 = output8.view(output8.size(0), -1)
        #output8 = self.probe8(output8)

        output = self.conv4_0(output)
        output9 = self.avg_pool(output)
        output9 = output9.view(output9.size(0), -1)
        #output9 = self.probe9(output9)

        output = self.conv4_1(output)
        output10 = self.avg_pool(output)
        output10 = output10.view(output10.size(0), -1)
        #output10 = self.probe10(output10)

        output = self.conv4_2(output)
        output11 = self.avg_pool(output)
        output11 = output11.view(output11.size(0), -1)
        #output11 = self.probe11(output11)

        output = self.conv4_3(output)
        output12 = self.avg_pool(output)
        output12 = output12.view(output12.size(0), -1)
        #output12 = self.probe12(output12)

        output = self.conv4_4(output)
        output13 = self.avg_pool(output)
        output13 = output13.view(output13.size(0), -1)
        #output13 = self.probe13(output13)

        output = self.conv4_5(output)
        output14 = self.avg_pool(output)
        output14 = output14.view(output14.size(0), -1)
        #output14 = self.probe14(output14)

        output = self.conv5_0(output)
        output15 = self.avg_pool(output)
        output15 = output15.view(output15.size(0), -1)
        #output15 = self.probe15(output15)

        output = self.conv5_1(output)
        output16 = self.avg_pool(output)
        output16 = output16.view(output16.size(0), -1)
        #output16 = self.probe16(output16)

        output = self.conv5_2(output)
        output17 = self.avg_pool(output)
        output17 = output17.view(output17.size(0), -1)
        #output17 = self.probe17(output17)

        # output = self.conv5_3(output)
        # output18 = self.avg_pool(output)
        # output18 = output18.view(output18.size(0), -1)
        # #output18 = self.probe18(output18)

        # output = self.conv5_4(output)
        # output19 = self.avg_pool(output)
        # output19 = output19.view(output19.size(0), -1)
        # #output19 = self.probe19(output19)

        # output = self.conv5_5(output)
        # output20 = self.avg_pool(output)
        # output20 = output20.view(output20.size(0), -1)
        # #output20 = self.probe20(output20)

        # output = self.conv5_6(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        # output18 = output.view(output.size(0), -1)

        output = self.fc(output)

        return output,[output1,output2,output3,output4,output5,output6,output7,output8,output9,output10,output11,output12,
        output13,output14,output15,output16,output17]

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

    return ResNet(BasicBlock, w)
    #return ResNet(BottleNeck, w)
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



