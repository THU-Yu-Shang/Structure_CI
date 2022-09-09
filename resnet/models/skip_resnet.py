import torch
import torch.nn as nn
import numpy as np 
import random

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """
    expansion = 1

    def __init__(self, in_channels, out_channels, skip_in_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
            #nn.ReLU(inplace=True),
            #nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        #shortcut
        self.shortcut = nn.Sequential()
        self.shortcut1 = nn.Sequential(
                    nn.Conv2d(256, out_channels , kernel_size=1, stride=int(out_channels/256), bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        #self.skip = skip
        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension

        if skip_in_channels > 0:
            if stride != 1 or skip_in_channels != BasicBlock.expansion * out_channels:
                if out_channels/skip_in_channels == 2:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(skip_in_channels, out_channels , kernel_size=1, stride=int(out_channels/skip_in_channels), bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
                if out_channels/skip_in_channels == 4:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(skip_in_channels, skip_in_channels*2 , kernel_size=1, stride=int(out_channels/skip_in_channels/2), bias=False),
                        nn.BatchNorm2d(skip_in_channels*2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(skip_in_channels*2, out_channels , kernel_size=1, stride=int(out_channels/skip_in_channels/2), bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
                    # self.shortcut = nn.Sequential(
                    #     nn.Conv2d(skip_in_channels, skip_in_channels*4 , kernel_size=1, stride=4, bias=False),
                    #     nn.BatchNorm2d(skip_in_channels*4),
                    # )
                if out_channels/skip_in_channels == 8:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(skip_in_channels, skip_in_channels*2 , kernel_size=1, stride=int(out_channels/skip_in_channels/4), bias=False),
                        nn.BatchNorm2d(skip_in_channels*2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(skip_in_channels*2, skip_in_channels*4, kernel_size=1, stride=int(out_channels/skip_in_channels/4), bias=False),
                        nn.BatchNorm2d(skip_in_channels*4),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(skip_in_channels*4, out_channels , kernel_size=1, stride=int(out_channels/skip_in_channels/4), bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
                    # self.shortcut = nn.Sequential(
                    #     nn.Conv2d(skip_in_channels, skip_in_channels*8 , kernel_size=1, stride=8, bias=False),
                    #     nn.BatchNorm2d(skip_in_channels*8),
                    # )
        

                print('run')

    def forward(self, x):

        if len(x[1]) > 1:
            # print(x[0].size())
            # print(x[1].size())
            # print(self.shortcut(x[1]).size())
            return nn.ReLU(inplace=True)(self.residual_function(x[0]) + self.shortcut(x[1]))
        else:
            return nn.ReLU(inplace=True)(self.residual_function(x[0]))

class BottleNeck(nn.Module):

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
        print(self.skip_weight)
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.skip_weight * self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_classes=100):
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.begin_out_channel = [64]*4+[128]*4+[256]*6+[512]*2
        self.skip_begin = list(np.arange(1,32,2))
        self.skip_end = []

        self.distance_p = -8

        for i in range(16):
            if self.skip_begin[i]+2<33:
                candidate = np.arange(self.skip_begin[i]+2,34,2)
                #print(candidate)
            else:
                candidate = np.array([33])
        # for i in range(16):
        #     if self.skip_begin[i] in [1,3,5]:
        #         candidate = np.arange(self.skip_begin[i]+2,8)
        #     if self.skip_begin[i] in [7,9,11,13]:
        #         candidate = np.arange(self.skip_begin[i]+2,16)
        #     if self.skip_begin[i] in [15,17,19,21,23,25]:
        #         candidate = np.arange(self.skip_begin[i]+2,28)
        #     if self.skip_begin[i] in [27,29]:
        #         candidate = np.arange(self.skip_begin[i]+2,34)
        #     if self.skip_begin[i] in [31]:
        #         candidate = np.array([33])
            pos = [float(candidate[j]-self.skip_begin[i])**self.distance_p for j in range(len(candidate))]
            pos = pos/np.sum(pos)
            N = np.random.rand()
            dic = {}
            #print(candidate)
            for n in range(len(pos)):
                    if n == 0:
                        dic[candidate[n]] = (0,pos[n])
                    else:
                        dic[candidate[n]] = (dic[candidate[n-1]][1],dic[candidate[n-1]][1]+pos[n])
                #判断随机数r落在哪一段
            for j in dic.keys():
                if N>dic[j][0] and N<= dic[j][1]:
                    # if i in [14,15]:
                    # Q = np.random.rand()
                    # if Q>1:
                    #     self.skip_end.append(33)
                    # else:
                        self.skip_end.append(j)
        # for b in range(1):
        #     self.skip_end[np.random.randint(0,16)] += 2
            #print(dic)
            #self.skip_end.append(random.randint(self.skip_begin[i]+2,33))
        #self.skip_end=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]
        print("skip end position:",self.skip_end)
        
        self.channel_setting = [0]*32
        for i in range(16):
            #print(self.skip_end[i]-2)
            self.channel_setting[self.skip_end[i]-2] = self.begin_out_channel[i]

        self.out_setting = [0]*32
        for i in range(16):
            self.out_setting[self.skip_end[i]-2] = self.skip_begin[i]

        self.conv2_0 = self._make_layer(block, 64, self.channel_setting[0], 1)
        self.conv2_1 = self._make_layer(block, 64, self.channel_setting[1], 1)
        self.conv2_2 = self._make_layer(block, 64, self.channel_setting[2], 1)
        self.conv2_3 = self._make_layer(block, 64, self.channel_setting[3], 1)
        self.conv2_4 = self._make_layer(block, 64, self.channel_setting[4], 1)
        self.conv2_5 = self._make_layer(block, 64, self.channel_setting[5], 1)

        self.conv3_0 = self._make_layer(block, 128, self.channel_setting[6], 2)
        self.conv3_1 = self._make_layer(block, 128, self.channel_setting[7], 1)
        self.conv3_2 = self._make_layer(block, 128, self.channel_setting[8], 1)
        self.conv3_3 = self._make_layer(block, 128, self.channel_setting[9], 1)
        self.conv3_4 = self._make_layer(block, 128, self.channel_setting[10], 1)
        self.conv3_5 = self._make_layer(block, 128, self.channel_setting[11], 1)
        self.conv3_6 = self._make_layer(block, 128, self.channel_setting[12], 1)
        self.conv3_7 = self._make_layer(block, 128, self.channel_setting[13], 1)

        self.conv4_0 = self._make_layer(block, 256, self.channel_setting[14], 2)
        self.conv4_1 = self._make_layer(block, 256, self.channel_setting[15], 1)
        self.conv4_2 = self._make_layer(block, 256, self.channel_setting[16], 1)
        self.conv4_3 = self._make_layer(block, 256, self.channel_setting[17], 1)
        self.conv4_4 = self._make_layer(block, 256, self.channel_setting[18], 1)
        self.conv4_5 = self._make_layer(block, 256, self.channel_setting[19], 1)
        self.conv4_6 = self._make_layer(block, 256, self.channel_setting[20], 1)
        self.conv4_7 = self._make_layer(block, 256, self.channel_setting[21], 1)
        self.conv4_8 = self._make_layer(block, 256, self.channel_setting[22], 1)
        self.conv4_9 = self._make_layer(block, 256, self.channel_setting[23], 1)
        self.conv4_10 = self._make_layer(block, 256, self.channel_setting[24], 1)
        self.conv4_11 = self._make_layer(block, 256, self.channel_setting[25], 1)

        self.conv5_0 = self._make_layer(block, 512, self.channel_setting[26], 2)
        self.conv5_1 = self._make_layer(block, 512, self.channel_setting[27], 1)
        self.conv5_2 = self._make_layer(block, 512, self.channel_setting[28], 1)
        self.conv5_3 = self._make_layer(block, 512, self.channel_setting[29], 1)
        self.conv5_4 = self._make_layer(block, 512, self.channel_setting[30], 1)
        self.conv5_5 = self._make_layer(block, 512, self.channel_setting[31], 1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)



    def _make_layer(self, block, out_channels, skip_in_channel, stride):

        layers = []
        layers.append(block(self.in_channels, out_channels, skip_in_channel, stride))
        self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):

        out_list = [0]*33
        # [output1, output2_0, output2_1,output2_2, output2_3, output2_4, output2_5,
        #             output3_0, output3_1, output3_2, output3_3, output3_4, output3_5, output3_6, output3_7,
        #             output4_0, output4_1, output4_2, output4_3, output4_4, output4_5, output4_6, output4_7, output4_8,output4_9, output4_10, output4_11, 
        #             output5_0, output5_1, output5_2, output5_3, output5_4, output5_5]

        o_setting = []
        for i in range(32): 
            if self.out_setting[i] > 0:
                o_setting.append(self.out_setting[i]-1)
            else:
                o_setting.append(-1)
        #print(o_setting)
        output1 = self.conv1(x)
        out_list[0] = output1 
        output2_0 = self.conv2_0([output1,out_list[o_setting[0]] if o_setting[0]>-1 else []])
        out_list[1] = output2_0
        output2_1 = self.conv2_1([output2_0,out_list[o_setting[1]] if o_setting[1]>-1 else []])
        out_list[2] = output2_1
        output2_2 = self.conv2_2([output2_1,out_list[o_setting[2]] if o_setting[2]>-1 else []])
        out_list[3] = output2_2
        output2_3 = self.conv2_3([output2_2,out_list[o_setting[3]] if o_setting[3]>-1 else []])
        out_list[4] = output2_3
        output2_4 = self.conv2_4([output2_3,out_list[o_setting[4]] if o_setting[4]>-1 else []])
        out_list[5] = output2_4
        output2_5 = self.conv2_5([output2_4,out_list[o_setting[5]] if o_setting[5]>-1 else []])
        out_list[6] = output2_5

        output3_0 = self.conv3_0([output2_5,out_list[o_setting[6]] if o_setting[6]>-1 else []])
        out_list[7] = output3_0
        output3_1 = self.conv3_1([output3_0,out_list[o_setting[7]] if o_setting[7]>-1 else []])
        out_list[8] = output3_1
        output3_2 = self.conv3_2([output3_1,out_list[o_setting[8]] if o_setting[8]>-1 else []])
        out_list[9] = output3_2
        output3_3 = self.conv3_3([output3_2,out_list[o_setting[9]] if o_setting[9]>-1 else []])
        out_list[10] = output3_3
        output3_4 = self.conv3_4([output3_3,out_list[o_setting[10]] if o_setting[10]>-1 else []])
        out_list[11] = output3_4
        output3_5 = self.conv3_5([output3_4,out_list[o_setting[11]] if o_setting[11]>-1 else []])
        out_list[12] = output3_5
        output3_6 = self.conv3_6([output3_5,out_list[o_setting[12]] if o_setting[12]>-1 else []])
        out_list[13] = output3_6
        output3_7 = self.conv3_7([output3_6,out_list[o_setting[13]] if o_setting[13]>-1 else []])
        out_list[14] = output3_7

        output4_0 = self.conv4_0([output3_7,out_list[o_setting[14]] if o_setting[14]>-1 else []])
        out_list[15] = output4_0
        output4_1 = self.conv4_1([output4_0,out_list[o_setting[15]] if o_setting[15]>-1 else []])
        out_list[16] = output4_1
        output4_2 = self.conv4_2([output4_1,out_list[o_setting[16]] if o_setting[16]>-1 else []])
        out_list[17] = output4_2
        output4_3 = self.conv4_3([output4_2,out_list[o_setting[17]] if o_setting[17]>-1 else []])
        out_list[18] = output4_3
        output4_4 = self.conv4_4([output4_3,out_list[o_setting[18]] if o_setting[18]>-1 else []])
        out_list[19] = output4_4
        output4_5 = self.conv4_5([output4_4,out_list[o_setting[19]] if o_setting[19]>-1 else []])
        out_list[20] = output4_5
        output4_6 = self.conv4_6([output4_5,out_list[o_setting[20]] if o_setting[20]>-1 else []])
        out_list[21] = output4_6
        output4_7 = self.conv4_7([output4_6,out_list[o_setting[21]] if o_setting[21]>-1 else []])
        out_list[22] = output4_7
        output4_8 = self.conv4_8([output4_7,out_list[o_setting[22]] if o_setting[22]>-1 else []])
        out_list[23] = output4_8
        output4_9 = self.conv4_9([output4_8,out_list[o_setting[23]] if o_setting[23]>-1 else []])
        out_list[24] = output4_9
        output4_10 = self.conv4_10([output4_9,out_list[o_setting[24]] if o_setting[24]>-1 else []])
        out_list[25] = output4_10
        output4_11 = self.conv4_11([output4_10,out_list[o_setting[25]] if o_setting[25]>-1 else []])
        out_list[26] = output4_11

        output5_0 = self.conv5_0([output4_11,out_list[o_setting[26]] if o_setting[26]>-1 else []])
        out_list[27] = output5_0
        output5_1 = self.conv5_1([output5_0,out_list[o_setting[27]] if o_setting[27]>-1 else []])
        out_list[28] = output5_1
        output5_2 = self.conv5_2([output5_1,out_list[o_setting[28]] if o_setting[28]>-1 else []])
        out_list[29] = output5_2
        output5_3 = self.conv5_3([output5_2,out_list[o_setting[29]] if o_setting[29]>-1 else []])
        out_list[30] = output5_3
        output5_4 = self.conv5_4([output5_3,out_list[o_setting[30]] if o_setting[30]>-1 else []])
        out_list[31] = output5_4
        output5_5 = self.conv5_5([output5_4,out_list[o_setting[31]] if o_setting[31]>-1 else []])
        out_list[32] = output5_5

        output = self.avg_pool(output5_5)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def resnet34():
    return ResNet(BasicBlock)
