import time
import numpy as np
import networkx as nx
from utils.dotdict import dotdict
import matplotlib.pyplot as plt
from numba import jit

import itertools
import multiprocessing

from utils.graph_generator import get_graph

from scripts.awareness.agent_country import Agent_Country

args = dotdict({
    "logfile": "log/temp.log",
    "plot": False,
    "max_iteration": 1000,
    "beta_super":0.0,
    "xi": 1,
    "p_teleport":0.0,
    "MAX_E_TIME":10,
    "MAX_I_TIME":10,
    "super_infected_agents": [],
    "p_super": 0.0,
    "awM": None,
    "awR": -1,
    "random_seed":0,
    "CPU_cores":2,
    "simnum":10,
})

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def get_diam(country, graph):
    Is = np.arange(len(graph))[country.states==2]
    #print(Is, graph.nodes)
    Is = [n for i,n in enumerate(graph.nodes) if country.states[i]==2]
    g = graph.subgraph(Is)
    
    nodes = sorted(nx.connected_components(g), key=len, reverse=True)[0]
    g = g.subgraph(nodes)

    ps = np.array(nx.get_node_attributes(g, 'pos').values())
    x1,x2,y1,y2 = [np.min(ps[:,0]), np.max(ps[:,0]), np.min(ps[:,1]), np.max(ps[:,1])]
    
    return nx.diameter(g), 

def run(args, beta, gamma, n, awM):
    args = dotdict(args)
    graph = get_graph("grid", {"n": n*n, "N": -1, "d": 4})
    args["I_time"]=int(1/gamma)
    args["infected_agents"]=[n//2+(n//2)*n]
    args["beta"]=beta
    args["gamma"]=gamma
    args["awM"] = awM


    res = np.zeros(shape=(args["max_iteration"], 4))
    diams = np.zeros(shape=(args["max_iteration"]))

    country = Agent_Country(args, graph)
    country.init_seeds = args["infected_agents"]
    Agent_Country.numba_random_seed(0)
    country.log_json()
    for i in range(args["max_iteration"]):
        res[i,2]=np.sum(country.states ==2)
        diams[i] = get_diam(country, graph)
        if country.check_stop():
            break
        country.step()
        country.log_json()

    return res[:i,2], diams[:i]

res,diams = run(args, 0.6, 0.55, 30, 0.1)

plt.plot(moving_average(res,7), c='b', label="Total infected( I)")
plt.plot(moving_average(diams,7), label="Diameter", c="r")
plt.legend()
plt.show()