import torch
import torch.nn as nn
import random
import numpy as np
from graph3 import RandomGraph
import math

# seed = 2 
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

def Merge(dict1, dict2): 
    res = {**dict1, **dict2} 
    return res 

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def graph_trans(o_nodes,o_in_edges):
        in_edges={}
        nodes = []
        for node in o_nodes:
            nodes.append(node)
            in_edges[node] = []
        for node in o_nodes:
            nodes.append(node+len(o_nodes))
            in_edges[node+len(o_nodes)] = []
        
        for i in o_nodes:
            for j in range(len(o_in_edges[i])):  #有向图
                p1 = random.uniform(0,1)
                p2 = random.uniform(0,1)
                #if p1 < 1:   
                #    in_edges[o_in_edges[i][j]+len(o_nodes)].append(i)
                if p2 < 1:  #被附庸节点指向附庸节点
                    in_edges[i+len(o_nodes)].append(o_in_edges[i][j])               
            in_edges[i+len(o_nodes)].append(i)  #identical link

        #fully-connected
        # for i in o_nodes:
        #     for j in range(len(o_nodes)):
        #         in_edges[i+len(o_nodes)].append(j)

        return nodes , in_edges

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

        # self.apply(weights_init)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise(x)
        return x


# ReLU-convolution-BN triplet
class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Unit, self).__init__()

        self.dropout_rate = 0.2

        self.unit = nn.Sequential(
            #nn.ReLU(),
            SeparableConv2d(in_channels, out_channels, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        return self.unit(x)


class Node(nn.Module):
    def __init__(self, itself, in_degree, in_channels, out_channels, stride=1):
        super(Node, self).__init__()
        self.in_degree = in_degree
        self.itself = itself
        #print(self.itself)
        self.weights = 0.5*torch.ones(len(self.in_degree))
        self.unit = Unit(in_channels, out_channels, stride=stride)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, *input):
        #print(self.weights)
        if self.itself < 0:
            if len(self.in_degree) > 1:
                #x = (input[0] * torch.sigmoid(self.weights[0]))
                x = input[0] * self.weights[0]
                for index in range(1, len(input)):
                    # x += (input[index] * torch.sigmoid(self.weights[index]))
                    x += input[index] * self.weights[index]
                if self.in_channels == self.out_channels+100:
                    out = x
                else:
                    out = self.unit(x)

                # different paper, add identity mapping
                # out += x
            else:
                if self.in_channels == self.out_channels+100:
                    out = input[0]
                else:
                    out = self.unit(input[0])
        else:
            if len(self.in_degree) > 1:
                if self.itself == 0:
                    #x = (input[0] * torch.sigmoid(torch.tensor(1)))
                    x = input[0]
                else:
                    #x = (input[0] * torch.sigmoid(self.weights[0]))
                    x = input[0] * self.weights[0]
                for index in range(1, len(input)):
                    #print(index)
                    #print('len',len(input))
                    if index == self.itself:
                        #x += (input[index] * torch.sigmoid(torch.tensor(1)))
                        x += input[index] 
                    else:
                        #x += (input[index] * torch.sigmoid(self.weights[index]))
                        #print(len(input))
                        #print(self.weights.size())
                        x += input[index] *self.weights[index]
                if self.in_channels == self.out_channels+100:
                    out = x
                else:    
                    out = self.unit(x)

                # different paper, add identity mapping
                # out += x
            else:
                if self.in_channels == self.out_channels+100:
                    out = input[0]
                else:
                    out = self.unit(input[0])
        temp = self.weights.detach().clone() 
        if self.itself > -1:
            temp[self.itself]= 1
        #print(temp)
        return out, temp 


class RandWire(nn.Module):
    def __init__(self, save, subgroup, start_flag, end_flag, node_num, modularity, G_shuffle, alpha, p, k, m,  in_channels, out_channels, graph_mode, is_train, name):
        super(RandWire, self).__init__()
        self.node_num = node_num
        self.p = p
        self.k = k
        self.m = m
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.is_train = is_train
        self.name = name
        self.start_flag = start_flag  #is start layer or not
        self.end_flag = end_flag 
        self.save = save
        self.subgroup = subgroup 
        self.d_list = []
        self.d_counter = np.zeros(self.node_num)
        self.d_counter2 = np.zeros(self.node_num)
        self.in_counter = np.zeros(self.node_num)
        self.out_weight = nn.Parameter(torch.ones(self.node_num), requires_grad=True) #最后一层aggregate各节点输出时的权重
        self.alpha = alpha
        self.M = modularity 
        self.G_shuffle = G_shuffle
        self.alpha_G = 1
        self.G_counter = np.zeros(self.M)
        self.G_list = []
        self.G_member = {}   #每个subgroup对应的节点
        self.adj = np.zeros((self.node_num,self.node_num))
        #subgroup-level BA network,M>1
        if self.M>1:
            G = RandomGraph(self.M, alpha=self.alpha_G, p=self.p, k=self.k, m=1,subgroup = 0,in_index = None,member= None, graph_mode=graph_mode)
            self.G_nodes, self.G_in_edges = G.make_graph()
            #print('before shuffle:',self.G_in_edges)
            for k in self.G_in_edges:
                self.G_list += self.G_in_edges[k]
            #print('G_list:',self.G_list)
            for j in range(len(self.G_list)):
                self.G_counter[self.G_list[j]] += 1  #不同subgroup各自的优先级
            #print('G_counter:',self.G_counter)
            re_index = list(reversed(sorted(range(len(self.G_counter)), key=lambda k: self.G_counter[k])))
            #random.shuffle(re_index)
            #print('before shuffle',re_index)
            
            G_ori_index = re_index.copy() 

            #shuffle G_level BA network
            G_shuffle_level = self.G_shuffle

            for k in range(G_shuffle_level):
                r1 = random.randint(0,self.M-1)
                r2 = random.randint(0,self.M-1)
                #print(r1)
                #print(r2)
                re_index[r1],re_index[r2] = re_index[r2],re_index[r1] 
            #print('after shuffle',re_index)
            
            G_shuffled_edges = {}
            x = 0
            for u in G_ori_index:
                G_shuffled_edges[re_index[x]] = []
                for r in self.G_in_edges[G_ori_index[x]]:
                    G_shuffled_edges[re_index[x]].append(re_index[G_ori_index.index(r)])
                x = x+1
            self.G_in_edges = G_shuffled_edges
            #print('after shuffle',self.G_in_edges)

            #重新统计
            self.G_counter = np.zeros(self.M)
            self.G_list = []
            for k in self.G_in_edges:
                self.G_list += self.G_in_edges[k]
            #print('G_list:',self.G_list)
            for j in range(len(self.G_list)):
                self.G_counter[self.G_list[j]] += 1  #不同subgroup各自的优先级
            #print('G_counter:',self.G_counter)
            re_index = list(reversed(sorted(range(len(self.G_counter)), key=lambda k: self.G_counter[k])))
            #print('priority',re_index)
            
            #生成dic：每个subgroup对应的成员
            
            c = 0 #start position
            for t in range(self.M):
                if t < self.M-1:
                    member_num = round(self.node_num/self.M)
                    #print(member_num)
                    self.G_member[t] = list(range(c,c+member_num))
                    c = c+member_num
                else:
                    member_num = self.node_num-(self.M-1)*round(self.node_num/self.M)
                    #print(member_num)
                    #print(list(range((r-1)*round(self.node_num/self.M),self.node_num)))
                    self.G_member[t] = list(range(c,c+member_num))
                    c = c+member_num
        else:
            self.G_member = {0:list(range(self.node_num))}
            re_index = [0]
            self.M=1
            self.G_in_edges= {0:[]}

        #print(self.G_member)

        #individual-level BA network
        #one grpah
        if self.subgroup == 1:
            graph_node = RandomGraph(self.node_num, alpha=self.alpha, p=self.p, k=self.k, m=self.m, subgroup =self.M, in_index=re_index, member=self.G_member, g_relation = self.G_in_edges, graph_mode=graph_mode)

        #merge N graph
        # if self.subgroup == 2:
        #     graph_node1 = RandomGraph(int(self.node_num/2), alpha=self.alpha, p=self.p, k=self.k, m=self.m, graph_mode=graph_mode)
        #     graph_node2 = RandomGraph(int(self.node_num/2), alpha=self.alpha, p=self.p, k=self.k, m=self.m, graph_mode=graph_mode)
        
        # if self.subgroup == 4:
        #     graph_node1 = RandomGraph(int(self.node_num/4), alpha=self.alpha, p=self.p, k=self.k, m=self.m, graph_mode=graph_mode)
        #     graph_node2 = RandomGraph(int(self.node_num/4), alpha=self.alpha, p=self.p, k=self.k, m=self.m, graph_mode=graph_mode)
        #     graph_node3 = RandomGraph(int(self.node_num/4), alpha=self.alpha, p=self.p, k=self.k, m=self.m, graph_mode=graph_mode)
        #     graph_node4 = RandomGraph(int(self.node_num/4), alpha=self.alpha, p=self.p, k=self.k, m=self.m, graph_mode=graph_mode)

        if self.is_train is True:
            print("is_train: True")
            #one graph
            if self.subgroup == 1:
                #graph = graph_node.make_graph()
                #self.o_nodes, self.o_in_edges = graph_node.get_graph_info(graph)
                self.o_nodes, self.o_in_edges = graph_node.make_graph()
                self.nodes, self.in_edges = graph_trans(self.o_nodes,self.o_in_edges)
                print(self.in_edges)

                #提取邻接矩阵
                for k in self.in_edges.keys():
                    if k > self.node_num-1:
                        for h in self.in_edges[k]:
                            self.adj[h,k-self.node_num] = 1 
                            # if h == k-self.node_num:
                            #     self.adj[h,k-self.node_num] += 1
                            # else:
                            #     self.adj[h,k-self.node_num] += 0.5
                print(self.adj)

                #print(self.nodes)
                #graph_node.save_random_graph(graph, name)

            #merge 2 graph
            # if self.subgroup == 2:
            #     g1 = graph_node1.make_graph()
            #     e1 = graph_node1.get_graph_info(g1)
            #     g2 = graph_node2.make_graph()
            #     e2 = graph_node2.get_graph_info(g2)
            #     self.o_nodes = e1[0]+list(np.array(e2[0])+int(self.node_num/2))
            #     #改key
            #     a = list(e2[1].keys()).copy()
            #     for k in a:
            #         e2[1][k+int(self.node_num/2)] = e2[1].pop(k)
            #     #改value
            #     for i in list(e2[1].keys()):
            #         for j in range(len(e2[1][i])):
            #             e2[1][i][j] += int(self.node_num/2)
            #     self.o_in_edges = Merge(e1[1],e2[1])
            #     self.nodes, self.in_edges = graph_trans(self.o_nodes,self.o_in_edges)

            # #graph = graph_node.load_random_graph(name)
            # #self.nodes, self.in_edges = graph_node.get_graph_info(graph)
            
            
            # #merge 4 graph
            # if self.subgroup == 4:
            #     g1 = graph_node1.make_graph()
            #     e1 = graph_node1.get_graph_info(g1)
            #     g2 = graph_node2.make_graph()
            #     e2 = graph_node2.get_graph_info(g2)
            #     g3 = graph_node3.make_graph()
            #     e3 = graph_node3.get_graph_info(g3)
            #     g4 = graph_node4.make_graph()
            #     e4 = graph_node4.get_graph_info(g4)
            #     self.o_nodes = e1[0]+list(np.array(e2[0])+int(self.node_num/4))+list(np.array(e3[0])+2*int(self.node_num/4))+list(np.array(e4[0])+3*int(self.node_num/4))
            #     #改key
            #     a = list(e2[1].keys()).copy()
            #     for k in a:
            #         e2[1][k+int(self.node_num/4)] = e2[1].pop(k)
            #     #改value
            #     for i in list(e2[1].keys()):
            #         for j in range(len(e2[1][i])):
            #             e2[1][i][j] += int(self.node_num/4)
            #     o_in_edges_1 = Merge(e1[1],e2[1])

            #     #改key
            #     a = list(e3[1].keys()).copy()
            #     for k in a:
            #         e3[1][k+2*int(self.node_num/4)] = e3[1].pop(k)
            #     #改value
            #     for i in list(e3[1].keys()):
            #         for j in range(len(e3[1][i])):
            #             e3[1][i][j] += 2*int(self.node_num/4)
            #     o_in_edges_2 = Merge(o_in_edges_1,e3[1])

            #     #改key
            #     a = list(e4[1].keys()).copy()
            #     for k in a:
            #         e4[1][k+3*int(self.node_num/4)] = e4[1].pop(k)
            #     #改value
            #     for i in list(e4[1].keys()):
            #         for j in range(len(e4[1][i])):
            #             e4[1][i][j] += 3*int(self.node_num/4)
            #     self.o_in_edges = Merge(o_in_edges_2,e4[1])
                
            #     self.nodes, self.in_edges = graph_trans(self.o_nodes,self.o_in_edges)
                #print(self.in_edges)

            for k in self.in_edges:
                #print(k)
                self.d_list += self.in_edges[k]
                #self.in_counter[k]
            #print(d_list)
            for j in range(len(self.d_list)):
                self.d_counter[self.d_list[j]] += 1  #不带权重的出度
            #print(self.d_counter)
 
            
        else:
            graph = graph_node.load_random_graph(name)
            self.o_nodes, self.o_in_edges = graph_node.get_graph_info(graph)
            self.nodes, self.in_edges = graph_trans(self.o_nodes,self.in_edges)
            #print(self.in_edges)
        
        # define input layer
        self.module_list = nn.ModuleList([Node(-1, self.in_edges[0], self.in_channels, self.out_channels, stride=2)])
        self.module_list.extend([Node(-1, self.in_edges[node], self.in_channels, self.out_channels, stride=2) for node in range(1,int(len(self.nodes)/2)) if node > 0])
        # define the output layer
        self.module_list.extend([Node(self.in_edges[node].index(node-int(self.node_num)),self.in_edges[node], self.out_channels, self.out_channels) for node in range(int(len(self.nodes)/2),int(len(self.nodes))) if node > 0])
    def forward(self, x):
        if not self.start_flag:
            #print('+1')
            memory = {}
            w_list = []
            #input nodes      
            for node in self.nodes[:int(len(self.nodes)/2)]:
                #print('second',node)
                #print(x[node].size())
                memory[node],_ = self.module_list[node].forward(x[node])
            for node in self.nodes[int(len(self.nodes)/2):int(len(self.nodes))]:
                # print(node, self.in_edges[node][0], self.in_edges[node])
                if len(self.in_edges[node]) > 1:
                    out, w = self.module_list[node].forward(*[memory[in_vertex] for in_vertex in self.in_edges[node]])
                    w_list += (list(w.cpu().numpy()))
                    #print([memory[in_vertex] for in_vertex in self.in_edges[node]][0].size())
                else:
                    out, w = self.module_list[node].forward(memory[self.in_edges[node][0]])
                    w_list += (list(w.cpu().numpy()))
                memory[node] = out

            self.d_counter2 = np.zeros(self.node_num)
            for j in range(len(self.d_list)):
                self.d_counter2[self.d_list[j]] += w_list[j]
 
                #print(memory[node].size())
            #print(memory[0].size())
            
            #out = memory[self.in_edges[self.node_num + 1][0]]
            #for in_vertex_index in range(1, len(self.in_edges[self.node_num + 1])):
            #    out += memory[self.in_edges[self.node_num + 1][in_vertex_index]]
            #    #print(self.in_edges[self.node_num + 1][in_vertex_index])
            #out = out / len(self.in_edges[self.node_num + 1])

            if not self.end_flag: 
                output = []
                for j in self.nodes[int(len(self.nodes)/2):int(len(self.nodes))]:
                    output.append(memory[j].unsqueeze(0))
                output = torch.cat(output,dim=0)    
                return output,self.d_counter,self.d_counter2,self.G_member,self.adj

            else:  #最后一个randwire模块
                output = memory[self.nodes[int(len(self.nodes)/2)]] * torch.sigmoid(self.out_weight[0])
                # if self.save:
                #     torch.save(output,'out_tensor3/0.pt')
                for i in self.nodes[int(len(self.nodes)/2)+1:int(len(self.nodes))]:
                    output += torch.sigmoid(self.out_weight[i-int(len(self.nodes)/2)])*memory[i]
                # if self.save:
                #         #print(self.nodes[int(len(self.nodes)/2):int(len(self.nodes))])
                #         np.save('out_tensor/agent.npy',memory)
                #output = output / (len(self.nodes)/2)
                return output,self.d_counter,self.d_counter2,memory,self.G_member,self.adj
        
        else: #第一个randwire模块，接收同一个input
            memory = {}
            w_list = []
            for node in self.nodes[:int(len(self.nodes)/2)]:
                memory[node],_ = self.module_list[node].forward(x)
            for node in self.nodes[int(len(self.nodes)/2):int(len(self.nodes))]:
                # print(node, self.in_edges[node][0], self.in_edges[node])
                #print('first',node)
                if len(self.in_edges[node]) > 1:
                    #print(self.in_edges[node])
                    out, w = self.module_list[node].forward(*[memory[in_vertex] for in_vertex in self.in_edges[node]])
                    
                    w_list += (list(w.cpu().numpy()))
                    #print([memory[in_vertex] for in_vertex in self.in_edges[node]][0].size())
                else:
                    out, w = self.module_list[node].forward(memory[self.in_edges[node][0]])
                    w_list += (list(w.cpu().numpy()))
                memory[node] = out
            #print(w_list)
            self.d_counter2 = np.zeros(self.node_num)
            for j in range(len(self.d_list)):
                self.d_counter2[self.d_list[j]] += w_list[j]
            
            output = []
            for j in self.nodes[int(len(self.nodes)/2):int(len(self.nodes))]:
                output.append(memory[j].unsqueeze(0))
            output = torch.cat(output,dim=0)            

            return output,self.d_counter,self.d_counter2,self.G_member,self.adj
            


