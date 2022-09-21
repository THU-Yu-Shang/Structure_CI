import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from randwire3 import RandWire
import seaborn
import numpy
import matplotlib.pyplot as plt
from gpu_mem_track import MemTracker

def cal_gini(d_counter,node_num):
    cum_degree = np.cumsum(sorted(np.append(d_counter, 0)))
    sum_degree = cum_degree[-1]
    xarray = np.array(range(0, len(cum_degree))) / np.float(len(cum_degree) - 1)
    yarray = cum_degree / sum_degree
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    G = A / (A + B)
    return G

def cal_metric(d_counter,node_num):
    sd = np.std(d_counter)
    mean = np.mean(d_counter)
    #print("sd:{0:.8f}, mean:{1:.8f}".format(sd, mean))

    cum_degree = np.cumsum(sorted(np.append(d_counter, 0)))
    sum_degree = cum_degree[-1]
    xarray = np.array(range(0, len(cum_degree))) / np.float(len(cum_degree) - 1)
    yarray = cum_degree / sum_degree
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    G = A / (A + B)
    #print("gini index:",G)

    E = 0
    for i in range(len(d_counter)):
        p = d_counter[i]/sum(d_counter)
        E += p*np.log(node_num*p)
    #print("entropy:",E)
    return sd, G, E

class Model(nn.Module):
    def __init__(self, node_num, modular, G_shuff, alpha, p, k, m, in_channels, out_channels, graph_mode, model_mode, dataset_mode, is_train):
        super(Model, self).__init__()
        self.node_num = node_num
        self.p = p
        self.k = k
        self.m = m
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.model_mode = model_mode
        self.is_train = is_train
        self.dataset_mode = dataset_mode
        self.alpha = alpha
        self.modularity = modular
        self.G_shuffle = G_shuff

        self.num_classes = 1000
        self.dropout_rate = 0.2

        if self.dataset_mode is "CIFAR10":
            self.num_classes = 10
        elif self.dataset_mode is "CIFAR100":
            self.num_classes = 100
        elif self.dataset_mode is "IMAGENET":
            self.num_classes = 1000
        elif self.dataset_mode is "MNIST":
            self.num_classes = 10

        if self.model_mode is "CIFAR10":
            self.CIFAR_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=self.out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU()
            )
            self.CIFAR_conv2 = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels),
            )
            # self.CIFAR_conv2 = nn.Sequential(
            #     RandWire(self.node_num, self.p, self.in_channels, self.out_channels, self.graph_mode, self.is_train, name="CIFAR10_conv2")
            # )
            self.CIFAR_conv3 = nn.Sequential(
                RandWire(self.node_num, self.p, self.in_channels, self.out_channels * 2, self.graph_mode, self.is_train, name="CIFAR10_conv3")
            )
            self.CIFAR_conv4 = nn.Sequential(
                RandWire(self.node_num, self.p, self.in_channels * 2, self.out_channels * 4, self.graph_mode, self.is_train, name="CIFAR10_conv4")
            )

            self.CIFAR_classifier = nn.Sequential(
                nn.Conv2d(self.in_channels * 4, 1280, kernel_size=1),
                nn.BatchNorm2d(1280)
            )
        elif self.model_mode is "CIFAR100":
            self.CIFAR100_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=self.out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels),
            )

            self.CIFAR100_conv2 = nn.Sequential(
                RandWire(self.node_num, self.p, self.in_channels, self.out_channels * 2, self.graph_mode, self.is_train, name="CIFAR100_conv2")
            )
            self.CIFAR100_conv3 = nn.Sequential(
                RandWire(self.node_num, self.p, self.in_channels * 2, self.out_channels * 4, self.graph_mode, self.is_train, name="CIFAR100_conv3")
            )
            self.CIFAR100_conv4 = nn.Sequential(
                RandWire(self.node_num, self.p, self.in_channels * 4, self.out_channels * 8, self.graph_mode, self.is_train, name="CIFAR100_conv4")
            )

            self.CIFAR100_classifier = nn.Sequential(
                nn.Conv2d(self.in_channels * 8, 1280, kernel_size=1),
                nn.BatchNorm2d(1280)
            )
        elif self.model_mode is "SMALL_REGIME":
            self.SMALL_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=self.out_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels // 2),
                nn.ReLU()
            )
            self.SMALL_conv2 = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels // 2, out_channels=self.out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels)
            )
            self.SMALL_conv3 = nn.Sequential(
                RandWire(False, 1, True, False, self.node_num, self.modularity, self.G_shuffle, self.alpha, self.p, self.k, self.m, self.in_channels, self.out_channels, self.graph_mode, self.is_train, name="SMALL_conv3")
            )
            self.SMALL_conv4 = nn.Sequential(
                RandWire(False, 1, False, False, self.node_num, self.modularity, self.G_shuffle, self.alpha, self.p, self.k, self.m,  self.in_channels, self.out_channels * 2, self.graph_mode, self.is_train, name="SMALL_conv4")
            )
            self.SMALL_conv5 = nn.Sequential(
                RandWire(False, 1, False, False, self.node_num, self.modularity, self.G_shuffle, self.alpha, self.p, self.k, self.m, self.in_channels * 2 , self.out_channels * 2, self.graph_mode, self.is_train, name="SMALL_conv5")
            )
            self.SMALL_conv6 = nn.Sequential(
               RandWire(False, 1, False, False, self.node_num, self.modularity, self.G_shuffle, self.alpha, self.p, self.k, self.m, self.in_channels * 2, self.out_channels * 4, self.graph_mode, self.is_train, name="SMALL_conv6")
            )
            self.SMALL_conv7 = nn.Sequential(
               RandWire(False, 1, False, False, self.node_num, self.modularity, self.G_shuffle, self.alpha, self.p, self.k, self.m, self.in_channels * 4, self.out_channels * 4, self.graph_mode, self.is_train, name="SMALL_conv7")
            )
            self.SMALL_conv8 = nn.Sequential(
               RandWire(False, 1, False, True, self.node_num, self.modularity, self.G_shuffle, self.alpha, self.p, self.k, self.m, self.in_channels * 4, self.out_channels * 8, self.graph_mode, self.is_train, name="SMALL_conv8")
            )
            self.SMALL_classifier = nn.Sequential(
                nn.Conv2d(self.in_channels * 8, 1280, kernel_size=1),
                nn.BatchNorm2d(1280)
            )
        elif self.model_mode is "REGULAR_REGIME":
            self.REGULAR_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=self.out_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels // 2)
            )
            self.REGULAR_conv2 = nn.Sequential(
                RandWire(self.node_num // 2, self.p, self.in_channels // 2, self.out_channels, self.graph_mode, self.is_train, name="REGULAR_conv2")
            )
            self.REGULAR_conv3 = nn.Sequential(
                RandWire(self.node_num, self.p, self.in_channels, self.out_channels * 2, self.graph_mode, self.is_train, name="REGULAR_conv3")
            )
            self.REGULAR_conv4 = nn.Sequential(
                RandWire(self.node_num, self.p, self.in_channels * 2, self.out_channels * 4, self.graph_mode, self.is_train, name="REGULAR_conv4")
            )
            self.REGULAR_conv5 = nn.Sequential(
                RandWire(self.node_num, self.p, self.in_channels * 4, self.out_channels * 8, self.graph_mode, self.is_train, name="REGULAR_conv5")
            )
            self.REGULAR_classifier = nn.Sequential(
                nn.Conv2d(self.in_channels * 8, 1280, kernel_size=1),
                nn.BatchNorm2d(1280)
            )

        self.output = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(1280, self.num_classes)
        )

    def forward(self, x):
        if self.model_mode is "CIFAR10":
            out = self.CIFAR_conv1(x)
            out = self.CIFAR_conv2(out)
            out = self.CIFAR_conv3(out)
            out = self.CIFAR_conv4(out)
            out = self.CIFAR_classifier(out)
        elif self.model_mode is "CIFAR100":
            out = self.CIFAR100_conv1(x)
            #print(out.size())
            out = self.CIFAR100_conv2(out)
            #print(out.size())
            out = self.CIFAR100_conv3(out)
            out = self.CIFAR100_conv4(out)
            out = self.CIFAR100_classifier(out)
        elif self.model_mode is "SMALL_REGIME":
            #print('b1')
            out = self.SMALL_conv1(x)
            #print('b2')
            out = self.SMALL_conv2(out)  
            #print('b3')    
            out,d_counter1,d_counter4,g_mem1,adj1 = self.SMALL_conv3(out)
            #print('b4')
            out,d_counter2,d_counter5,g_mem2,adj2 = self.SMALL_conv4(out)
            #print('b5')
            out,d_counter3,d_counter6,g_mem3,adj3 = self.SMALL_conv5(out)
            #print(out.size())
            out,d_counter7,d_counter8, g_mem4,adj4 = self.SMALL_conv6(out)
            out,d_counter9,d_counter10, g_mem5,adj5 = self.SMALL_conv7(out)
            out,d_counter11,d_counter12, memory, g_mem6, adj6 = self.SMALL_conv8(out)
            out = self.SMALL_classifier(out)
            #print(out.size())

            #d_counter = d_counter1 + d_counter2 + d_counter3 + d_counter7
            #d_counter = d_counter1 + d_counter2 + d_counter3
            #d_counter0 = d_counter4 + d_counter5 + d_counter6 + d_counter8
            d_counter0 = d_counter4 + d_counter5 + d_counter6 + d_counter8 +d_counter10 +d_counter12
            # print("no-weighted:")
            # sd1,g1,e1 = cal_metric(d_counter,self.node_num)
            # #print(d_counter)
            # sd_a1,g_a1,e_a1 = cal_metric(d_counter1,self.node_num)
            # sd_a2,g_a2,e_a2 = cal_metric(d_counter2,self.node_num)
            # sd_a3,g_a3,e_a3 = cal_metric(d_counter3,self.node_num)
            # print('no weight avg sd in different rounds:',(sd_a1+sd_a2+sd_a3)/3)
            # print('no weight avg gini index in different rounds:',(g_a1+g_a2+g_a3)/3)
            # print('no weight avg entropy in different rounds:',(e_a1+e_a2+e_a3)/3)
            # print('no weight sumed sd:',sd1)
            # print('no weight sumed gini index:',g1)
            # print('no weight sumed entropy:',e1)

            #print(sum(d_counter))
            # with open("./m4_sumed1.5_gini_range1.txt", "a") as f1:
            #     f1.write(str(g1))
            #     f1.write("\n")
            print("weighted:")
            sd_avg = 0
            g_avg = 0
            sd_total = []
            g_total = []
            for t in g_mem1.keys():  #每个subgroup
                #print(t)
                mem_num = len(g_mem1[t])
                for u in range(mem_num):  #一个subgroup的每个个体
                    degree_list = []
                    degree_list.append(d_counter4[g_mem1[t][u]])
                    degree_list.append(d_counter5[g_mem2[t][u]])
                    degree_list.append(d_counter6[g_mem3[t][u]])
                    degree_list.append(d_counter8[g_mem4[t][u]])
                    degree_list.append(d_counter10[g_mem5[t][u]])
                    degree_list.append(d_counter12[g_mem6[t][u]])
                    sd_ind, g_ind, e_ind = cal_metric(np.array(degree_list),len(degree_list))
                    print((degree_list))
                    sd_total.append(sd_ind)
                    g_total.append(g_ind)
                    sd_avg += sd_ind
                    g_avg += g_ind
            sd_avg = sd_avg/self.node_num
            g_avg = g_avg/self.node_num
            sd_median = (sorted(sd_total)[int(len(sd_total)/2)] + sorted(sd_total)[int(len(sd_total)/2)+1])/2
            g_median = (sorted(g_total)[int(len(g_total)/2)] + sorted(g_total)[int(len(g_total)/2)+1])/2

            # fig,axes=plt.subplots(1,2, figsize=(16, 6))
            # plt.subplot(1,2,1)
            # seaborn.distplot(sd_total,ax=axes[0])
            # plt.title('surprise index distribution(sd)',fontsize = 24)
            # plt.xlabel('Standard Deviation',fontsize = 20)
            # plt.ylabel('Density',fontsize = 20)
            # plt.subplot(1,2,2)
            # seaborn.distplot(g_total,ax=axes[1])
            # plt.title('surprise index distribution(gini index)',fontsize = 24)
            # plt.xlabel('Gini Index',fontsize = 20)
            # plt.ylabel('Density',fontsize = 20)
            # #plt.subplots_adjust(wspace=0.5)
            # plt.savefig('surprise_index_plot2/surprise_index_distribution_M'+str(len(g_mem1.keys()))+'_pG1_pI'+str(self.alpha)+'.png',dpi=100)

            #print('avg surprise index(sd):',sd_avg)
            #print('avg surprise index(gini index):',g_avg)
            #print('median avg surprise index(sd):',sd_median)
            #print('median surprise index(gini index):',g_median)
            #print('surprise index data(sd):',sd_total)
            #print('surprise index data(gini index):',g_total)

            sd2,g2,e2 = cal_metric(d_counter0,self.node_num)
            sd_a4,g_a4,e_a4 = cal_metric(d_counter4,self.node_num)
            sd_a5,g_a5,e_a5 = cal_metric(d_counter5,self.node_num)
            sd_a6,g_a6,e_a6 = cal_metric(d_counter6,self.node_num)
            sd_a8,g_a8,e_a8 = cal_metric(d_counter8,self.node_num)
            sd_a10,g_a10,e_a10 = cal_metric(d_counter8,self.node_num)
            sd_a12,g_a12,e_a12 = cal_metric(d_counter12,self.node_num)
            #print('with weight avg sd in different rounds:',(sd_a4+sd_a5+sd_a6)/3)
            print('with weight avg gini index in different rounds(giniA):',(g_a4+g_a5+g_a6+g_a8+g_a10+g_a12)/6)
            # print(g_a4)
            # print(g_a5)
            # print(g_a6)
            # print(g_a8)
            # print(g_a10)
            # print(g_a12)
            #print('with weight avg entropy in different rounds:',(e_a4+e_a5+e_a6)/3)
            #print('with weight sumed sd:',sd2)
            print('with weight sumed gini index(giniB):',g2)
            #print('with weight sumed entropy:',e2)
            # with open("./m4_sumed1.5_gini_range2.txt", "a") as f2:
            #     f2.write(str(g2))
            #     f2.write("\n")

            '''calculate num of paths'''
            path1 = adj1@adj2@adj3@adj4@adj5@adj6
            value1 = np.sum(path1, axis=1)
            print('num of paths:',np.sum(value1))

            # '''calculate balance index(undirected graph)'''
            # adj_matrix = np.zeros(adj1.shape)
            # adj_matrix = adj1+adj1.T+adj2+adj2.T+adj3+adj3.T+adj4+adj4.T+adj5+adj5.T+adj6+adj6.T-6*np.eye(12)
            # print('accumulated link distribution:\n',adj_matrix)
            # out_d = []
            # for i in range(adj_matrix.shape[0]):  
            #     out1 = np.sum(adj_matrix[i,:])
            #     out_d.append(out1)
            # out_d.sort()
            # # print(out_d)
            # front = out_d[:int(len(out_d)/2)]  #小
            # back = out_d[int(len(out_d)/2):]  #大
            # total_g = cal_gini(out_d,len(out_d))
            # #front_g = cal_gini(front,len(front))
            # back_g = cal_gini(back,len(back)) #hub node
            # print('front:',front)
            # print('back:',back)
            # #print('front gini index:',front_g)
            # print('back gini index:',back_g)
            # print('total gini index:',total_g)
            # print('balance index:',total_g/back_g)

            adj_matrix = np.zeros(adj1.shape)
        elif self.model_mode is "REGULAR_REGIME":
            out = self.REGULAR_conv1(x)
            out = self.REGULAR_conv2(out)
            out = self.REGULAR_conv3(out)
            out = self.REGULAR_conv4(out)
            out = self.REGULAR_conv5(out)
            out = self.REGULAR_classifier(out)

        # global average pooling
        batch_size, channels, height, width = out.size()
        out = F.avg_pool2d(out, kernel_size=[height, width])
        # out = F.avg_pool2d(out, kernel_size=x.size()[2:])
        out = torch.squeeze(out)
        out = self.output(out)

        return out,sd2,e2,g2,memory 
