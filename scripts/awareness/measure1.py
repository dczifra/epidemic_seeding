import time
import numpy as np
import networkx as nx
from utils.dotdict import dotdict
import matplotlib.pyplot as plt

import itertools
import multiprocessing

from utils.graph_generator import get_graph

from scripts.awareness.agent_country import Agent_Country

args = dotdict({
    "logfile": "/tmp/temp.log",
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
    "CPU_cores":4,
    "simnum":100,
})

def run(args, beta, gamma, n, awM):
    args = dotdict(args)
    graph = get_graph("grid", {"n": n*n, "N": -1, "d": 4})
    args["I_time"]=int(1/gamma)
    args["infected_agents"]=[n//2+(n//2)*n]
    args["beta"]=beta
    args["gamma"]=gamma
    args["awM"] = awM

    country = Agent_Country(args, graph)
    history = country.run_fast(0, args.infected_agents, simnum=args["simnum"])
    return history

def multi_run(args, betas, gammas, ns, awMs):
    temp_res = {}
    pool = multiprocessing.Pool(processes=args.CPU_cores)

    for (beta, gamma), n, awM in itertools.product(zip(betas, gammas), ns, awMs):
        temp_res[(beta, gamma, n, awM)]=pool.apply_async(run, args=(dict(args), beta, gamma, n, awM))

    pool.close()
    pool.join()

    res = {}
    for params, hist in temp_res.items():
        res[params] = hist.get()

    return res

def measure0():
    betas = np.array([0.6,0.7])
    res = multi_run(args, betas, betas/2, [20,30], [0.1])

    # Measure 0: Infected agents
    for params, hist in res.items():
        y = np.mean(hist, axis=0, keepdims=False)[:,2]

        plt.plot(y)
        plt.title("beta,gamma,n = "+str(params))
        plt.show()

def measure1(betas, gammas, ns, awMs, xs, plot_args):
    res = multi_run(args, betas, gammas, ns, awMs)

    res_y = {p: np.mean(hist[:,args.max_iteration//2:,2], axis=(0,1), keepdims=False) for p,hist in res.items()}
    res_err = {p: np.std(hist[:,args.max_iteration//2:,2], axis=(0,1), keepdims=False) for p,hist in res.items()}

    ys = [res_y[(beta,gamma,n,awM)] for (beta,gamma),n,awM in itertools.product(zip(betas,gammas),ns,awMs)]
    y_err = [res_err[(beta,gamma,n,awM)] for (beta,gamma),n,awM in itertools.product(zip(betas,gammas),ns,awMs)]

    # === Plot ===
    plt.errorbar(np.log(betas),ys, yerr=y_err)
    plt.title(plot_args["title"])
    plt.xlabel(plot_args["xlabel"])
    plt.ylabel(plot_args["ylabel"])

    sl, inter = np.polyfit(xs,ys, 1)
    plt.legend(["slope {:.3f} intercept {:.3f}".format(sl, inter)])
    plt.show()

    return ys, y_err

def table1():
    # === 1 ===
    betas = np.array(range(1,10))/10
    gammas = betas/2

    measure1(betas, gammas, [10], [0.1],
            xs = np.log(betas),
            plot_args={
                "xlabel":"log(beta)",
                "ylabel":"I_s",
                "title":"(a) gamma=beta/2 ,n=10, m=0.1"})

    # === 2 ===
    betas = np.array(range(1,10))/10
    gb=list(map(lambda x: np.exp(x), range(1,10)))
    gammas = betas/np.array(gb)

    measure1(betas, gammas, [10], [0.1],
            xs = np.log(betas/gammas),
            plot_args={
                "xlabel":"log(beta/gamma)",
                "ylabel":"I_s",
                "title":"(a) gamma=beta/2 ,n=10, m=0.1"})

if(__name__ == "__main__"):
    #measure0()
    table1()

