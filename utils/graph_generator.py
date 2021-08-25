import csv
import numpy as np
import networkx as nx
import powerlaw
import pickle

#import matplotlib.pyplot as plt

def get_graph(type, args):
    """
    Returns the type of graph with the given size
    :param type: The graph's type you wish (simple, erdos-renyi ...)
    :param size: Node size of th graph
    """
    if type == "simple":
        return get_simple()
    elif type == "read_from":
        return read_gml(args)
    elif type == "erdos-renyi":
        return get_erdos_renyi(args)
    elif type == "linklist":
        return get_linklist(args)
    elif type =="nx-pickle":
        return get_pickle(args)
    elif type == "KSH":
        return get_KSH(args)
    elif type == "megye":
        return get_linklist_megye(args)
    elif type =="random-regular":
        return get_random_regular(args)
    elif type =="random-geometric":
        return get_random_geometric(args)
    elif type =="pref_attachment":
        return get_pref_attachment(args)
    elif type =="config_model":
        return get_config_model(args)
    elif type =="single-node":
        return get_single_node(args)
    elif type =="path":
        return get_path(args)
    elif type =="grid":
        return get_grid(args)
    elif type =="2grids":
        return get_2grids(args)
    elif type =="torus":
        return get_torus(args)
    else:
        print("Type : {} not recognized".format(type))

def read_gml(args):
    graph = nx.Graph()
    return nx.read_graphml(args["path"])

def get_simple():
    graph = nx.Graph()

    graph.add_node("Budapest", index = 0, population=2000000)
    graph.add_node("Gyor",  index = 1, population=200000)
    graph.add_node("Szekesfehervar", index = 2, population=500000)
    graph.add_node("Kecskemet", index = 3, population=300000)
    graph.add_node("Pecs", index = 4, population=120000)

    graph.add_edge("Budapest", "Gyor", weight=100)
    graph.add_edge("Budapest", "Szekesfehervar", weight=30000)
    graph.add_edge("Budapest", "Kecskemet", weight=5000)
    graph.add_edge("Szekesfehervar", "Gyor", weight=50)
    graph.add_edge("Pecs", "Gyor", weight=500)


    print("Nodes: {}".format(graph.nodes()))
    print("Edges: {}".format(graph.edges()))

    nx.write_graphml(graph, "data/simple_graph.gml")
    nx.read_graphml("data/simple_graph.gml")

    return graph

def get_path(args):
    graph = nx.Graph()
    pop = args["population"]
    travel = args["population"]//2

    graph.add_node(0,index = 0, population = pop, pos = (0,0))
    for i in range(1,args["n"]):
        graph.add_node(i,index = i, population = pop, pos = (i,0))
        graph.add_edge(i, i-1, weight = travel, dist = 1.0)

    #nx.draw(graph)
    #plt.show()
    return graph

def get_erdos_renyi(args, plot=False):
    n = args["n"]
    p = args["d"]/n

    G = nx.erdos_renyi_graph(n, p)

    # === Init node population ===
    #print("  ==> Erdos-Renyi Graph")
    for i,v in enumerate(nx.nodes(G)):
        G.nodes[v]["population"]=max(1, nx.degree(G,v)*150)
        G.nodes[v]["index"]=i
        #print('%s %d %f' % (v, nx.degree(G, v), nx.clustering(G, v)))

    # === Init edge weights ===
    for u,v in nx.edges(G):
        G[u][v]["weight"]=np.min([30, nx.degree(G,u),nx.degree(G,v) ])
        G[u][v]["dist"] = 1.0

    pos = nx.spring_layout(G)
    nx.set_node_attributes(G, {k:[v[0]*15,v[1]*15] for k,v in  pos.items() }, 'pos')

    #if(plot):
    #    nx.draw(G)
    #    plt.show()
    return G

def get_pref_attachment(args, plot=False):
    n = args["n"]
    
    G = nx.barabasi_albert_graph(n, 3)

    # === Init node population ===
    #print("  ==> Erdos-Renyi Graph")
    for i,v in enumerate(nx.nodes(G)):
        G.nodes[v]["population"]=max(1, nx.degree(G,v)*150)
        G.nodes[v]["index"]=i
        #print('%s %d %f' % (v, nx.degree(G, v), nx.clustering(G, v)))

    # === Init edge weights ===
    for u,v in nx.edges(G):
        G[u][v]["weight"]=30.0
        G[u][v]["dist"] = 1.0

    pos = nx.spring_layout(G)
    nx.set_node_attributes(G, {k:[v[0]*15,v[1]*15] for k,v in  pos.items() }, 'pos')

    #if(plot):
    #    nx.draw(G)
    #    plt.show()
    return G


def get_config_model(args, plot=False):
    n = args["n"]
    if "deg_exp" in args:
        deg_exp = args["deg_exp"]
    else:
        deg_exp = 3.5
        
    d=powerlaw.Power_Law(xmin=3,parameters=[deg_exp]).generate_random(n).astype(int)
    if (sum(d) %2)>0:
        d[-1]+=1
    
    G = nx.configuration_model(d)
    G=nx.Graph(G)

    # === Init node population ===
    #print("  ==> Erdos-Renyi Graph")
    for i,v in enumerate(nx.nodes(G)):
        G.nodes[v]["population"]=max(1, nx.degree(G,v)*150)
        G.nodes[v]["index"]=i
        #print('%s %d %f' % (v, nx.degree(G, v), nx.clustering(G, v)))

    # === Init edge weights ===
    for u,v in nx.edges(G):
        G[u][v]["weight"]=30.0
        G[u][v]["dist"] = 1.0

    pos = nx.spring_layout(G)
    nx.set_node_attributes(G, {k:[v[0]*15,v[1]*15] for k,v in  pos.items() }, 'pos')

    #if(plot):
    #    nx.draw(G)
    #    plt.show()
    return G


def get_random_regular(args):
    n = args["n"]
    d = args["d"]
    N = args["N"]

    #m = args["m"]
    #G = nx.gnm_random_graph(n, m)
    G = nx.random_regular_graph(d,n)

    # some properties
    for i,v in enumerate(nx.nodes(G)):
        G.nodes[v]["population"]=int(N/n)
        G.nodes[v]["index"]=i
        #print('%s %d %f' % (v, nx.degree(G, v), nx.clustering(G, v)))

    for u,v in nx.edges(G):
        G[u][v]["weight"]=int(N/(n*d))
        G[u][v]["dist"] = 1.0
    
    pos = nx.spring_layout(G)
    nx.set_node_attributes(G, {k:[v[0]*15,v[1]*15] for k,v in  pos.items() }, 'pos')
    nx.set_node_attributes(G, {n:n for n in G.nodes() }, 'label')

    return G

def get_random_geometric(args):
    n = args["n"]
    d = args["d"]
    N = args["N"]

    #m = args["m"]
    #G = nx.gnm_random_graph(n, m)
    G = nx.random_geometric_graph(n,np.sqrt(d/(np.pi*n)))

    # some properties
    for i,v in enumerate(nx.nodes(G)):
        G.nodes[v]["population"]=int(N/n)
        G.nodes[v]["index"]=i
        #print('%s %d %f' % (v, nx.degree(G, v), nx.clustering(G, v)))

    for u,v in nx.edges(G):
        G[u][v]["weight"]=int(N/(n*d))
        G[u][v]["dist"] = 1.0
    
    pos = nx.get_node_attributes(G,"pos")
    nx.set_node_attributes(G, {k:[v[0]*15,v[1]*15] for k,v in  pos.items() }, 'pos')
    nx.set_node_attributes(G, {n:n for n in G.nodes() }, 'label')
    return G



def get_grid(args):
    n = args["n"]
    d = args["d"]
    N = args["N"]
    #print(n)

    #m = args["m"]
    #G = nx.gnm_random_graph(n, m)
    G = nx.grid_2d_graph(int(np.sqrt(n)),int(np.sqrt(n)))

    # some properties
    for i,v in enumerate(nx.nodes(G)):
        G.nodes[v]["population"]=int(N/G.number_of_nodes())
        G.nodes[v]["index"]=i
        #print('%s %d %f' % (v, nx.degree(G, v), nx.clustering(G, v)))

    for u,v in nx.edges(G):
        G[u][v]["weight"]=int(N/(G.number_of_nodes()*d))
        G[u][v]["dist"]=1.0
        
    nx.set_node_attributes(G, {n:n for n in G.nodes() }, 'pos')
    nx.set_node_attributes(G, {n:n for n in G.nodes() }, 'label')

    return G


def get_2grids(args):
    n = args["n"]
    d = args["d"]
    N = args["N"]
    #print(n)
    n=int(np.sqrt(n))

    #m = args["m"]
    #G = nx.gnm_random_graph(n, m)
    G = nx.grid_2d_graph(n,n)

    rows=range(n)
    columns=range(n)
    G.add_nodes_from( (i+n+1,j) for i in rows for j in columns )
    G.add_edges_from( ((i+n+1,j),(i-1+n+1,j)) for i in rows for j in columns if i>0 )
    G.add_edges_from( ((i+n+1,j),(i+n+1,j-1)) for i in rows for j in columns if j>0 )


    # some properties
    for i,v in enumerate(nx.nodes(G)):
        G.nodes[v]["population"]=int(N/G.number_of_nodes())
        G.nodes[v]["index"]=i
        #print('%s %d %f' % (v, nx.degree(G, v), nx.clustering(G, v)))

    for u,v in nx.edges(G):
        G[u][v]["weight"]=int(N/(G.number_of_nodes()*d))
        G[u][v]["dist"]=1.0
        
    nx.set_node_attributes(G, {n:n for n in G.nodes() }, 'pos')
    nx.set_node_attributes(G, {n:n for n in G.nodes() }, 'label')

    return G


def get_torus(args):
    n = args["n"]
    d = args["d"]
    N = args["N"]
    #print(n)

    #m = args["m"]
    #G = nx.gnm_random_graph(n, m)
    G = nx.grid_2d_graph(int(np.sqrt(n)),int(np.sqrt(n)), periodic=True)

    # some properties
    for i,v in enumerate(nx.nodes(G)):
        G.nodes[v]["population"]=int(N/G.number_of_nodes())
        G.nodes[v]["index"]=i
        #print('%s %d %f' % (v, nx.degree(G, v), nx.clustering(G, v)))

    for u,v in nx.edges(G):
        G[u][v]["weight"]=int(N/(G.number_of_nodes()*d))
        G[u][v]["dist"]=1.0
        
    nx.set_node_attributes(G, {n:n for n in G.nodes() }, 'pos')
    nx.set_node_attributes(G, {n:n for n in G.nodes() }, 'label')

    return G


def get_single_node(args):
    G = nx.Graph()
    G.add_node(0)
    G.nodes[0]["population"]=args["N"]
    G.nodes[0]["index"]=0
    return G
    
    
def get_pickle(args):
    with open(args["path"], 'rb') as handle:
        return pickle.load(handle)


def get_linklist(args):
    graph = nx.Graph()

    # === Read Nodes ===
    with open(args["nodes"], newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader) # Skip header
        for i,row in enumerate(spamreader):
            pop = int(row[2])
            graph.add_node(int(row[0]), index=int(row[0]), city_name = row[1], label = row[1],
                population = pop, pos = (float(row[4])*10, float(row[3])*10))
    
    # === Read Edges ===
    with open(args["edges"], newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader) # Skip header
        for i,row in enumerate(spamreader):
            edge_weight = 42*float(row[2])
            graph.add_edge(int(row[0]), int(row[1]), weight = edge_weight, dist = 1.0)
    
    # === Drop out small cities ===
    population = nx.get_node_attributes(graph, "population")
    big_cities = [i for i in graph.nodes if population[i]>10000]
    
    graph = graph.subgraph(big_cities)

    # === Get largest connected component ===
    graph = graph.subgraph(next(nx.connected_components(graph)))
    for i,n in enumerate(graph.nodes): # We need to reindex
        graph.nodes[n]["index"]=i
    print("Number of cities: ", graph.number_of_nodes())
    
    return graph

def get_KSH(args):
    th = {}
    if 'thresholds' not in args:
        th['node'] = 10000
        th['edge'] = 100
        th['node_factor'] = 1
        th['edge_factor'] = 1
    else:
        th = args['thresholds']
    
    if args["directed"]:
        graph0 = nx.DiGraph()
        print("Directed")
    else:
        graph0 = nx.Graph()

    # === Read Nodes ===
    indexes = {}
    with open(args["nodes"], newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader) # Skip header
        for i,row in enumerate(spamreader):
            pop = int(row[2])
            indexes[row[0]]=len(indexes)
            
            graph0.add_node(indexes[row[0]], index=indexes[row[0]],
                            city_name = row[1], label = row[1],
                            population = pop, pos = (float(row[4])*10, float(row[3])*10))
    
    nodes = graph0.nodes()
    # === Read Edges ===
    with open(args["edges"], newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader) # Skip header
        for i,row in enumerate(spamreader):
            #print(row[0], row[2])
            if((row[0] not in indexes) or  (row[2] not in indexes) or row[0]==row[2]):
                # One of the nodes is not in Hungary
                #print("Edge not found", row[0], row[2])
                continue
                
            edge_weight = float(row[4]) # All commuting
            graph0.add_edge(indexes[row[0]], indexes[row[2]],
                           weight = edge_weight, work = float(row[5]), school = float(row[6]), dist = 1.0)
    
    # === Drop out small cities ===
    population = nx.get_node_attributes(graph0, "population")
    big_cities = [node for node in graph0.nodes() if population[node]>th['node']]
    graph0 = graph0.subgraph(big_cities)

    # === Drop out small edges ===
    weights = nx.get_edge_attributes(graph0, "weight")
    big_edges = [edge for edge in graph0.edges() if weights[edge]>th['edge']]
    graph0 = graph0.edge_subgraph(big_edges)

    # Refactor graph edges
    for u,v in graph0.edges():
        graph0[u][v]["weight"]/= th['edge_factor']
    # Refactor node size
    for node in graph0.nodes:
        graph0.nodes[node]["population"]//=th['node_factor']
    
    # === Get largest connected component ===
    #graph = nx.connected_components(graph)[0]
    if args['directed']:
        nodes = sorted(nx.weakly_connected_components(graph0), key = len, reverse=True)[0]
    else:
        nodes = sorted(nx.connected_components(graph0), key=len, reverse=True)[0]
    graph0 = graph0.subgraph(nodes)

    # === Population cannot be larger, than the sum of neighbours
    for node in graph0.nodes():
        pop = graph0.nodes[node]['population']
        neigh_num = np.sum([graph0[node][n]['weight'] for n in graph0.neighbors(node)]) 
        if(pop < neigh_num):
            graph0.nodes[node]['population'] = int(neigh_num)+1
    
    population = nx.get_node_attributes(graph0, "population")
    # === Equal population ===
    if(("equal_pop" in args) and args["equal_pop"]): 
        avg_pop = int(np.mean(list(population.values())))
        for n in graph0.nodes:
            graph0.nodes[n]["population"] = avg_pop

    # === Equal edge weights ===
    if(("equal_edge" in args) and args["equal_edge"]):
        weights = nx.get_edge_attributes(graph0, "weight")
        avg_weight = float(np.mean(list(weights.values())))
        for u,v in graph0.edges:
            graph0[u][v]["weight"] = avg_weight

    # === Shuffle edges===
    if(args["directed"]):
        graph0 = nx.DiGraph(graph0)
    else:
        graph0 = nx.Graph(graph0)
    
    if(("shuffle_edges" in args) and args["shuffle_edges"]):
        avg_weight = float(np.mean(list(weights.values())))
        n = nx.algorithms.swap.connected_double_edge_swap(graph0, nswap=10000)
        print("Shuffled edges", n)
        # === set weights ===
        for u,v in graph0.edges:
            graph0[u][v]["weight"] = avg_weight

    new_keys = {k:i for i,k in enumerate(graph0.nodes())}
    for i,n in enumerate(graph0.nodes): # We need to reindex
        graph0.nodes[n]["index"]=i
    
    print("Number of cities: ", graph0.number_of_nodes())
    return nx.relabel_nodes(graph0, new_keys, copy=False)
    
def get_linklist_megye(args):
    graph = nx.Graph()

    # === Read Nodes ===
    with open("public_data/megye.map", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader) # Skip header
        for i,row in enumerate(spamreader):
            pop = int(row[2])
            graph.add_node(int(row[0]), index=int(row[0]), city_name = row[1], label = row[1],
                           population = pop, pos = (float(row[4])*2, float(row[3])*2), dist = 1.0)
    # === Read Edges ===
    with open("public_data/megye.edges", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader) # Skip header
        for i,row in enumerate(spamreader):
            graph.add_edge(int(row[0]), int(row[1]), weight = 1.0, dist = 1.0)

    for i,n in enumerate(graph.nodes): # We need to reindex
        graph.nodes[n]["index"]=i
        
    return graph
