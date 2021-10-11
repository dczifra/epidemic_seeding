import numpy as np
import networkx as nx

class TreelikeHierarchicalGraph:
    def extend_leafes(self, graph, begin_leaves, end_leaves):
        k = self.k
        N = len(graph)

        for x in range(end_leaves-begin_leaves+1):
            root = begin_leaves+x
            first = N+k*x
            for j1 in range(k):
                x1 = first+j1
                graph.add_edge(root,x1)
                
                if(self.mode == "cycle"):
                    if(j1+1 < k):
                        graph.add_edge(x1,x1+1)
                    else:
                        graph.add_edge(x1, first)
                elif(self.mode == "dense"):
                    for j2 in range(j1+1,k):
                        x2 = first+j2
                        graph.add_edge(x1,x2)
                elif(self.mode == "empty"):
                    pass
                else:
                    print("Mode not found")

        return graph,N,N+(end_leaves-begin_leaves+1)*k-1

    def __init__(self, rec_num, head, k, mode = "dense"):
        graph = nx.Graph()
        graph.add_node(0)
        a,b = 0,0
        self.k = k
        self.mode = mode

        shells = [[0]]
        for i in range(rec_num):
            graph,a,b = self.extend_leafes(graph, a, b)
            shells.append(list(range(a,b+1)))

        if(head != None):
            n = head["n"]
            for i in range(int(head["p_centrum"]*n*n/2)):
                u,v = np.random.randint(0,n,2)
                graph.add_edge(u,v)

            N = len(graph)
            for i in range(int(head["p_all"]*N*N/2)):
                u,v = np.random.randint(0,N,2)
                graph.add_edge(u,v)

        self.graph = graph
        self.shells = shells


class SimpleHierarchicalGraph:
    def get(self, depth, k, mode = "empty"):
        if(depth==0):
            G = nx.Graph()
            G.add_node(0)
            return G,[0]
        elif(depth == 1):
            G = nx.Graph()
            for i in range(1,k):
                G.add_edge(0,i)
            
            if(mode == "cycle"):
                for i in range(1,k):
                    if(i+1 < k):
                        G.add_edge(i, (i+1))
                G.add_edge(k-1,1)
            elif(mode == "dense"):
                for i in range(1,k):
                    for j in range(i+1,k):
                        G.add_edge(i,j)
            elif(mode == "empty"):
                pass
            else:
                print("Mode not found")
            return G, list(range(1,k))
        else:
            g, per = self.get(depth-1,k, mode)
            graphs = [g for i in range(k)]
            n = len(graphs[0])
            G = nx.algorithms.operators.all.disjoint_union_all(graphs)
            
            per_all = []
            for i in range(1,k):
                for node in per:
                    G.add_edge(0,i*n+node)
                    per_all.append(i*n+node)
            
            return G,per_all
            
            
            
            
