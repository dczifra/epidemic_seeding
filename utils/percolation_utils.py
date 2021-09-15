import os
import random
import pickle
import numpy as np
import multiprocessing

import powerlaw
import networkx as nx

from utils.params import init_graph, get_centrum
from utils.ecl_utils import run_ecl_cc,feed_edge_list, get_graphstream

def get_pc(graph):
    a=list(dict(graph.degree).values())
    Sum1=0
    Sum2=0
    for i in a:
        Sum2+=i*(i-1)
        Sum1+=i
    
    return Sum1/Sum2
    
def get_area_orig_ind_top_bottom(g,location,n):
    if(location == "top"):
        return list(map(lambda x: x[0], sorted(g.degree, key=lambda x: x[1], reverse=True)))[:n]
    elif(location == "bottom"):
        area = list(map(lambda x: x[0], sorted(g.degree, key=lambda x: x[1], reverse=True)))[-n:]
        return np.random.choice(area, size = n, replace = False)

def get_config_model(seed, n, deg_exp, folder):        
    random.seed(seed)
    np.random.seed(seed)
    
    d=powerlaw.Power_Law(xmin=3,parameters=[deg_exp]).generate_random(n).astype(int)
    if (sum(d) %2)>0:
        d[-1]+=1
    
    G = nx.configuration_model(d)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    nx.set_edge_attributes(G, 1.0, "weight")
    
    #centrum = get_centrum(G, "k-core", len(G.nodes))
    centrum = get_area_orig_ind_top_bottom(G,"top",len(G.nodes))
    g_stream = get_graphstream(G, centrum, args={"verbose":False})
    p_c = str(get_pc(G))
    with open("{}/pop_{}/config_{}_cen({}).pickle".format(folder,n, deg_exp, seed), "wb") as file:
        file.write(pickle.dumps(g_stream+"#"+p_c))
 
    #per = list(G.nodes())
    #g_stream = get_graphstream(G, per, inp_args)
    #with open("data/pop_{}/config_{}_per({}).pickle".format(n, deg_exp, seed), "wb") as file:
    #    pickle.dump(g_stream, file)
    return "Done",G,p_c

def print_random(seed,n,deg_exp):
    random.seed(seed)
    np.random.seed(seed)
    
    d=powerlaw.Power_Law(xmin=3,parameters=[deg_exp]).generate_random(n).astype(int)
    return seed, d

def init_graphs_parallel(pop_size,deg_exp, folder="data"):
    if(not os.path.exists("{}/pop_{}".format(folder, pop_size))):
        os.mkdir("{}/pop_{}".format(folder, pop_size))
    
    if(pop_size < 11000000):
        num_network = 10
        processes = 10 if pop_size < 5100000 else 3
    else:
        num_network = 3
        processes = 1
        
    pool = multiprocessing.Pool(processes=processes)
    res = []
    for i in range(num_network):
        filename = "{}/pop_{}/config_{}_{}({}).pickle".format(folder, pop_size, deg_exp, "cen", i)
        if(os.path.exists(filename)):
            op = type("MyOptionParser", (object,), {"get": lambda self: "Exists" })
            res.append(op())
        else:
            r = pool.apply_async(get_config_model, args = (i, pop_size, deg_exp, folder))
            res.append(r)
    
    res = [r.get() for r in res]
    return res

def load_graph(pop_size, deg_exp, random_seed, cen_type, folder="data"):
    filename = "{}/pop_{}/config_{}_{}({}).pickle".format(folder, pop_size, deg_exp, cen_type, random_seed)
    
    g_stream, p_c = pickle.load(open(filename, "rb")).split('#')
    return g_stream, float(p_c)


