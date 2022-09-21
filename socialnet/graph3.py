import networkx as nx
import os
from ba3 import ba_graph

class RandomGraph(object):
    def __init__(self, node_num, alpha=1, p=0.75, k=4, m=5, subgroup=1,in_index = None,member =None,g_relation =None, graph_mode="BA"):
        self.node_num = node_num
        self.p = p
        self.k = k
        self.m = m
        self.graph_mode = graph_mode
        self.alpha = alpha
        self.subgroup = subgroup
        self.in_index = in_index
        self.member = member
        self.g_relation = g_relation

    def make_graph(self):
 
        if self.graph_mode is "ER":
            graph = nx.random_graphs.erdos_renyi_graph(self.node_num, self.p)
            print("p=",self.p)
        elif self.graph_mode is "WS":
            graph = nx.random_graphs.connected_watts_strogatz_graph(self.node_num, self.k, self.p)
            print("k=",self.k)
            print("p=",self.p)
        elif self.graph_mode is "BA":
            print(self.m)
            #graph = ba_graph(n = self.node_num, m = self.m, alpha = self.alpha)
            nodes,in_edges = ba_graph(n = self.node_num, m = self.m, alpha = self.alpha, ns = 26, subgroup = self.subgroup,in_index = self.in_index,member =self.member, g_relation = self.g_relation)
            #print("m=",self.m)
            print("alpha=",self.alpha)
            print("subgroup=",self.subgroup)

        #return graph
        return nodes,in_edges

    def get_graph_info(self,graph):
        in_edges = {}
        nodes = []
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            neighbors.sort()
            edges = []
            check = []
            for neighbor in neighbors:
                if node > neighbor:
                    edges.append(neighbor)
                    check.append(neighbor)
            in_edges[node] = edges
            nodes.append(node)
        return nodes, in_edges

    def save_random_graph(self, graph, path):
        if not os.path.isdir("saved_graph"):
            os.mkdir("saved_graph")
        nx.write_yaml(graph, "./saved_graph/" + path)

    def load_random_graph(self, path):
        return nx.read_yaml("./saved_graph/" + path)
