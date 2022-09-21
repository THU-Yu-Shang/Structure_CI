import networkx as nx
from networkx.utils import py_random_state
from networkx.utils import nodes_or_number
import numpy as np
import random
import math


@nodes_or_number(0)
def empty_graph(n=0, create_using=None, default=nx.Graph):

    if create_using is None:
        G = default()
    elif hasattr(create_using, "_adj"):
        # create_using is a NetworkX style Graph
        create_using.clear()
        G = create_using
    else:
        # try create_using as constructor
        G = create_using()

    n_name, nodes = n
    G.add_nodes_from(nodes)
    return G

def _random_subset(seq, m, rng):
    """ Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = []
    while len(targets) < m:
        x = rng.choice(seq)
        targets.append(x)
    return targets

#@py_random_state(2)
def ba_graph(n, m, alpha, ns, subgroup, in_index, member,g_relation, seed=np.random.RandomState(2)):
    if subgroup<1:  #group level
        if m < 1 or m >= n:
            raise nx.NetworkXError(
                f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
            )
        targets = list(range(m))
        counter = np.zeros(n)
        pro = np.zeros(n)
        # Start adding the other n-m nodes. The first node is m.
        source = m
        in_edges = {}
        nodes = []
        # for i in range(m):
        #     in_edges[i] = []
        #     nodes.append(i)
        for i in range(m):
            in_edges[i] = []
            for j in range(i):
                in_edges[i].append(j)
            nodes.append(i)
        while source < n:
            nodes.append(source)
            #G.add_edges_from(zip([source] * m, targets))
            in_edges[source] = targets
            for i in range(m):
                counter[targets[i]] += 1
            counter[source] += m
            #repeated_nodes.extend(targets)
            # And the new node "source" has m edges to add to the list.
            #print(n)
            #print(m)
            #print(alpha)
            targets = []
            for a in range(m):
                pro = counter**alpha
                pro = pro/sum(pro)   #每个节点被选中的概率
                #print(pro)
                index = np.where(pro>0)[0] #可能被选择的节点编号
                #每段区间生成
                r = random.random()
                dic = {}
                for i in range(len(index)):
                    if i == 0:
                        dic[index[i]] = (0,pro[i])
                    else:
                        dic[index[i]] = (dic[index[i-1]][1],dic[index[i-1]][1]+pro[index[i]])
                #判断随机数r落在哪一段
                for j in dic.keys():
                    if r>dic[j][0] and r<= dic[j][1]:
                        targets.append(j)

                #targets = _random_subset(repeated_nodes, m, seed)
            #print(targets)
            #print(dic)
            source += 1
        return nodes,in_edges #隐式定义
        '''
        #显式保证按节点度降序排列
        d_list = []
        d_counter = np.zeros(len(nodes))
        for k in in_edges:
            d_list += in_edges[k]
        for j in range(len(d_list)):
            d_counter[d_list[j]] += 1  #不带权重的出度
        re_index = list(reversed(sorted(range(len(d_counter)), key=lambda k: d_counter[k])))
        new_dic = {}
        for i in in_edges.keys():
            new_dic[i] = []
            for j in range(len(in_edges[re_index[i]])):
                new_dic[i].append(re_index.index(in_edges[re_index[i]][j]))
        
        return nodes,new_dic
        '''

        '''
        #打乱顺序hub节点前后不一致
        d_list = []
        d_counter = np.zeros(len(nodes))
        for k in in_edges:
            d_list += in_edges[k]
        for j in range(len(d_list)):
            d_counter[d_list[j]] += 1  #不带权重的出度
        re_index = list(reversed(sorted(range(len(d_counter)), key=lambda k: d_counter[k])))
        #random.shuffle(re_index)
        #print(re_index)
        #随机挑选两个元素交换位置，执行ns次
        for k in range(ns):
            r1 = random.randint(0,n-1)
            r2 = random.randint(0,n-1)
            #print(r1)
            #print(r2)
            re_index[r1],re_index[r2] = re_index[r2],re_index[r1] 
        new_dic = {}
        for i in in_edges.keys():
            new_dic[i] = []
            for j in range(len(in_edges[re_index[i]])):
                new_dic[i].append(re_index.index(in_edges[re_index[i]][j]))
        
        return nodes,new_dic
        '''
    else:  #individual level
        if m < 1 or m >= n:
            raise nx.NetworkXError(
                f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
            )
        #targets = list(range(m))
        counter = np.zeros(n)
        seen = np.zeros(n)
        #angle = list(np.zeros(n))
        pro = np.zeros(n)
        #source = m
        in_edges = {}
        nodes = []
        # for i in range(m):
        #     in_edges[i] = []
        #     nodes.append(i)

        #similarity
        # e = 0
        # for s in range(subgroup):
        #     for u in range(int(n/subgroup)):
        #         angle[e] = random.uniform(s*2/subgroup,(s+1)*2/subgroup)
        #         e = e+1
        #print('g_relation',g_relation)
        #先加入优先级最高的组的m个节点
        for i in range(m):
            in_edges[member[in_index[0]][i]] = []
            for j in range(i):
                in_edges[member[in_index[0]][i]].append(member[in_index[0]][j])
            nodes.append(member[in_index[0]][i])
            counter[member[in_index[0]][i]] += m-1
        #print('in_index',in_index)

        q = 0
        for h in range(len(in_index)):  #遍历每个subgroup
            #print('h',h)
            seen = np.zeros(n)
            if q==0:
                m_index = member[in_index[h]][m:]  #待加入节点集合
                if len(g_relation[in_index[h]] ) == 0:
                    seen[member[in_index[h]]] = 1
                else:
                    seen[member[in_index[h]]] = 1
                    for t in g_relation[in_index[h]]:
                        seen[member[t]] = 1
                #print(seen)
            else:
                m_index = member[in_index[h]]
                if len(g_relation[in_index[h]] ) == 0:
                    seen[member[in_index[h]]] = 1
                else:
                    seen[member[in_index[h]]] = 1
                    for t in g_relation[in_index[h]]:
                        seen[member[t]] = 1
                #print(seen)
            #print(m_index)
            for source in m_index:
                nodes.append(source)
                targets = []
                for a in range(m):
                    pro = counter**alpha  #popularity
                    pro = pro*seen
                    #print(pro)
                    pro = pro/sum(pro)   #每个节点被选中的概率
                    #print(pro)
                    index = np.where(pro>0)[0] #可能被选择的节点编号
                    #print(pro)
                    #每段区间生成
                    r = random.random()
                    dic = {}
                    for i in range(len(index)):
                        if i == 0:
                            dic[index[i]] = (0,pro[index[i]])
                        else:
                            dic[index[i]] = (dic[index[i-1]][1],dic[index[i-1]][1]+pro[index[i]])
                    #判断随机数r落在哪一段
                    for j in dic.keys():
                        if r>dic[j][0] and r<= dic[j][1]:
                            targets.append(j)
                            counter[j] += 1
                        #print(r)
                        #print(dic)
                counter[source] += m
                in_edges[source] = targets
            #return nodes,in_edges
            q = q+1
        #print('ori',in_edges)

        #组内shuffle
        total_shuffle_index = []
        ori_index = []

        I_shuffle_level = [1,1,1,1]  #within-group shuffle程度

        for h in range(len(in_index)):  #遍历每个subgroup
            m_sub = member[in_index[h]]  #成员编号
            ori_index += m_sub
            shuffle_index = m_sub.copy()      
            for k in range(I_shuffle_level[h]):
                r1 = random.randint(0,len(m_sub)-1)
                r2 = random.randint(0,len(m_sub)-1)
                #print(r1)
                #print(r2)
                shuffle_index[r1],shuffle_index[r2] = shuffle_index[r2],shuffle_index[r1] 
            total_shuffle_index += shuffle_index

        shuffled_edges = {}
        #print(total_shuffle_index)
        s = 0
        for co in range(len(ori_index)):
            shuffled_edges[total_shuffle_index[s]] = []
            for r in in_edges[ori_index[s]]:
                shuffled_edges[total_shuffle_index[s]].append(total_shuffle_index[ori_index.index(r)])
            s = s+1
        #print('shuffled',shuffled_edges)
        return total_shuffle_index,shuffled_edges

        # while source < n:
        #     nodes.append(source)
        #     #G.add_edges_from(zip([source] * m, targets))
        #     in_edges[source] = targets
        #     for i in range(m):
        #         #print(targets)
        #         counter[targets[i]] += 1
                
        #     counter[source] += m
        #     #repeated_nodes.extend(targets)
        #     # And the new node "source" has m edges to add to the list.
        #     #print(n)
        #     #print(m)
        #     #print(alpha)
        #     # if (source+1)!=n: 
        #     #     similar = [math.exp(-abs(angle[source+1]-i)) if abs(angle[source+1]-i)<1 else math.exp(-abs(angle[source+1]+2-i)) for i in angle]
        #         #print(similar)
        #     # else:
        #     #     break
        #     targets = []
        #     for a in range(m):
        #         pro = counter**alpha  #popularity
        #         pro = pro/sum(pro)   #每个节点被选中的概率
        #         #pro_sim = np.array(similar)  #pop*sim
        #         #pro_sim = pro_sim/sum(pro_sim)
        #         #pro = pro_pop*pro_sim
        #         #pro = pro/sum(pro)
        #         #print(pro)
        #         index = np.where(pro>0)[0] #可能被选择的节点编号
        #         #每段区间生成
        #         r = random.random()
        #         dic = {}
        #         for i in range(len(index)):
        #             if i == 0:
        #                 dic[index[i]] = (0,pro[i])
        #             else:
        #                 dic[index[i]] = (dic[index[i-1]][1],dic[index[i-1]][1]+pro[index[i]])
        #         #判断随机数r落在哪一段
        #         for j in dic.keys():
        #             if r>dic[j][0] and r<= dic[j][1]:
        #                 targets.append(j)
        #                 #print(j)

        #         #targets = _random_subset(repeated_nodes, m, seed)
        #     #print(targets)
        #     #print(dic)
        #     source += 1

        #return nodes,in_edges #隐式定义  

        # #按subgroup进行shuffle
        # d_list = []
        # d_counter = np.zeros(len(nodes))
        # for k in in_edges:
        #     d_list += in_edges[k]
        # for j in range(len(d_list)):
        #     d_counter[d_list[j]] += 1  #不带权重的出度
        # new_index = []
        # for t in range(subgroup):
        #     new_index = new_index+member[in_index[t]]
        #     #print(in_index)
    
        # new_dic = {}
        # r = 0
        # for i in new_index:
        #     new_dic[i] = []
        #     for j in range(len(in_edges[r])):
        #         new_dic[i].append(new_index[in_edges[r][j]])
        #     r = r+1
        # print(in_edges)
        # print(new_dic)
        # return nodes,new_dic



    