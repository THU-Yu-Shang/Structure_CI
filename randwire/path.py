import numpy as np
import copy
def find_all_path(graph, start, end):
        p = [[start, ]]
        pd = []
        while len(p):  # 仍有潜在可行路径
            new_p = []
            for path in p:
                #print(path)
                del_index = []
                node_now = path[-1]  # 取出最后一个点
                if node_now == end:  # 到达终点
                    pd.append(path)  # 存储完成路径
                    del_index.append(p.index(path))
                    #p.pop(p.index(path))  # 清理
                else:
                    adjacent_nodes = np.where(graph[node_now, :])
                    adjacent_nodes = adjacent_nodes[0]
                    adjacent_nodes = adjacent_nodes.tolist()
                    for i in range(len(adjacent_nodes) - 1, -1, -1):  # 刨除重复
                        if adjacent_nodes[i] in path:
                            adjacent_nodes.pop(i)

                    if len(adjacent_nodes) == 0:
                        p.pop(p.index(path))  # 清理

                    if len(path) > 34:  # 用于速度优化，节点数少时没用
                        p.pop(p.index(path))  # 清理

                    for node in adjacent_nodes:  # 尝试所有可行可能性
                        temp = copy.deepcopy(path)
                        temp.append(node)
                        new_p.append(temp)
            del_index.reverse()  # 对索引进行反转，使其从后往前删除
            for i in del_index:
                p.pop(i)
            p = new_p
           # print(p)
            print(len(pd))
            if len(pd) > 3000000:
                print(len(pd))
                return pd
        
        #print(len(pd))
        return pd
G = np.loadtxt('./adj_randwire/WS6_1.txt',delimiter=',')
path = find_all_path(G,0,33)
# path