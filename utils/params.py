import numpy as np
import networkx as nx
from utils.dotdict import dotdict
from utils.graph_generator import get_graph

GIRG_args1 = dotdict({
    "name":"GIRG",
    "N":1000,
    "tau":2.5,
    "alpha":2.3,
    "C_1":0.8,
    "whished_edgenum":3000,
    "pop_in_city":2000,
    "random_seed":0,
    "verbose":False
})

GIRG_args2 = dotdict({
    "name":"GIRG",
    "N":1000,
    "tau":3,
    "alpha":1.3,
    "C_1":0.8,
    "whished_edgenum":3000,
    "pop_in_city":2000,
    "random_seed":0,
    "verbose":False
})

GIRG_args3 = dotdict({
    "name":"GIRG",
    "N":1000,
    "tau":3.5,
    "alpha":1.3,
    "C_1":0.8,
    "whished_edgenum":3000,
    "pop_in_city":2000,
    "random_seed":0,
    "verbose":False
})

GIRG_args4 = dotdict({
    "name":"GIRG",
    "N":1000,
    "tau":3.5,
    "alpha":2.3,
    "C_1":0.8,
    "whished_edgenum":3000,
    "pop_in_city":2000,
    "random_seed":0,
    "verbose":False
})

def get_moving(graph, procent):
    population = nx.get_node_attributes(graph, "population")
    population =  sum(list(population.values()))
    edges = list(nx.get_edge_attributes(graph, "weight").values())
    p_M = procent/(np.sum(edges)/population)
    return p_M

def find_GIRG_with_edge_num(args, lower, upper):
    np.random.seed(args.random_seed)
    girg = GIRG(vertex_size=args.N,                     
            alpha=args.alpha,
            tau=args.tau,
            C_1=args.C_1,
            C_2=(lower+upper)/2,
            expected_weight=1,
            on_torus=False)

    # === Count edges in largest connected component ===
    nodes = sorted(nx.connected_components(girg.graph), key=len, reverse=True)[0]
    graph = girg.graph.subgraph(nodes)
    edge_num = len(graph.edges())
    # === Update threshhold after edge_num ===
    if(edge_num == args.whished_edgenum or upper-lower<0.0001):
        return girg
    else:
        if(edge_num> args.whished_edgenum):
            upper = (lower+upper)/2
        elif(edge_num < args.whished_edgenum):
            lower = (lower+upper)/2
        return find_GIRG_with_edge_num(args, lower, upper)


def init_graph(args):
    if(args.name == "KSH"):
        folder = os.path.dirname(os.path.abspath(__file__))+'/../'
        node_file = folder+"data/KSHSettlList_settlID_settlname_pop_lat_lon.csv"
        edge_file = folder+"data/KSHCommuting_c1ID_c1name_c2ID_c2name_comm_school_work_DIR.csv"
        if(!os.path.exists(node_file) or !os.path.exists(edge_file)):
            print(node_file, "or", edge_file, "doues not exists. Please ask the authors for permission.")
        
        graph = get_graph(type = "KSH",
                        args = {"nodes":node_file,
                                "edges":edge_file,
                                "thresholds":args.th,
                                "equal_pop":args.transform["equal_pop"],
                                "equal_edge":args.transform["equal_edge"],
                                "shuffle_edges":args.transform["shuffle_edges"],
                                "directed":args.directed if "directed" in args else True,
                        })
    elif(args.name == "GIRG"):
        if(("whished_edgenum" not in args) or ("random_seed" not in args)):
            print("Please give the edgenum and random_seed")
            exit(1)

        population_per_city = args.pop_in_city

        girg = find_GIRG_with_edge_num(args, 0,10) # Take care of the upper bound
        nx.set_node_attributes(girg.graph, {i:tuple(girg.vertex_locations[i]) for i in girg.graph.nodes()}, "pos")
        nx.set_node_attributes(girg.graph, population_per_city, "population")

        # === Get largest component ===
        nodes = sorted(nx.connected_components(girg.graph), key=len, reverse=True)[0]
        graph = girg.graph.subgraph(nodes)
        nx.set_node_attributes(graph, {n:i for i,n in enumerate(graph.nodes())}, "index")
        
        # === Set weights ===
        max_weight = max(list(graph.degree(weight="weight")), key=lambda l: l[1])[1]
        sum_weight = np.mean(list(nx.get_edge_attributes(graph, "weight").values()))
        moving_ratio = population_per_city/max_weight
        args["edge_weigth"] = sum_weight*moving_ratio-0.000001
        
        # === Swap edges ===
        graph0 = nx.Graph(graph)
        if(("config_model" in args) and args.config_model):
            swap_num = len(graph.edges())*10
            n = nx.algorithms.swap.connected_double_edge_swap(graph0, nswap=swap_num)
            if ("verbose" not in args) or args.verbose:
                print("Shuffled edges", n)
            graph = graph0
        
        # === Set weights and population ===
        for e in graph.edges():
            graph[e[0]][e[1]]["weight"]=args.edge_weigth
        for n in graph.nodes():
            graph.nodes[n]["population"]=args.pop_in_city

        new_keys = {k:i for i,k in enumerate(graph.nodes())}    
        graph = nx.relabel_nodes(graph, new_keys, copy=True)


    else:
        print("Graph mode not understood", args.graph)

    # === Nodes and population info ===
    population=nx.get_node_attributes(graph, "population")
    
    if ("verbose" not in args) or args.verbose:
        print("Graph nodes: ", graph.number_of_nodes()) 
        print("Poulation: ", sum(list(population.values())))

    return graph

def get_centrum(graph, mode, size):
    if(mode=="degree"):
        degs = graph.degree()
        nodes = sorted(graph.nodes(), key=lambda l: degs[l], reverse = True)
        return nodes[0:size]
    elif(mode=="k-core"):
        k_num = nx.algorithms.core.core_number(graph)
        
        nodes = sorted(graph.nodes(), key=lambda l: k_num[l], reverse=True)
        return nodes[0:size]
   