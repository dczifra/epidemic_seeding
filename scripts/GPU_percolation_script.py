import json
import random
import numpy as np
import pandas as pd

import pickle
from utils.ecl_utils import run_ecl_cc,feed_edge_list, get_graphstream
from utils.percolation_utils import init_graphs_parallel, get_config_model, load_graph

def ger_args():
    import argparse
    parser = argparse.ArgumentParser(description='Process input parameters')
    parser.add_argument('--tau', dest='tau', type=float, default=3.5,
                    help='network degree dispribution parameter (powerlaw)')
    parser.add_argument('--n', dest='n', type=int, default=100000,
                    help='network size')
    parser.add_argument('--seed', dest='random_seed', type=int, default=0,
                    help='Random seed')
    parser.add_argument('--sim_num', dest='sim_num', type=int, default=1000,
                    help='Number of simulations')
    parser.add_argument('--N_ps', dest='N_ps', type=int, default=20,
                    help='p scaling parameter')
    parser.add_argument('--N_ss', dest='N_ss', type=int, default=20,
                    help='s scaling parameter')
    parser.add_argument('--log_folder', dest='log_folder', type=str, default="../GPU_percolation/data",
                    help='log folder')
    args = parser.parse_args()
    print(args)
    return vars(args)

def save_mtx(mtx, mtx_cen, mtx_per, ps, ss, n, exp, folder, p_c, extra):
    df = pd.DataFrame()
    
    for pi,p in enumerate(ps):
        for si,s in enumerate(ss):
            df=df.append({"rat0":mtx[pi][si], "rat":(mtx_cen[pi][si], mtx_per[pi][si]),
                          #"cen_vals":mtx_cen[pi][si], "per_vals":mtx_per[pi][si],
                          "tau":exp, "n":n, "p":p, "s":s, "p_c":p_c},
                         ignore_index=True)
    
    df.to_csv("{}/grid_{}_{}_{}.csv".format(folder,extra, n,exp))

def get_ss(Ns, n):
    ss = (n**np.linspace(0.0,0.9,Ns)).astype(int)
    ss=ss[ss<n/4]
    ss = np.array(sorted(set(ss)), dtype=np.int32)
    
    return str(len(ss))+" "+" ".join([str(s) for s in ss])

def get_pc(graph):
    a=list(dict(graph.degree).values())
    Sum1=0
    Sum2=0
    for i in a:
        Sum2+=i*(i-1)
        Sum1+=i
    
    return Sum1/Sum2
    
def get_ps(p_c, tau, n, N):
    ps = p_c +  n**np.linspace(-np.abs(tau-3)/(tau-1)*2,-0.05, N)
    return str(len(ps))+" "+" ".join([str(s) for s in ps])

def save_to_common_file(params):
    df_all = pd.DataFrame()

    n =params["n"]
    tau = params["tau"]

    files = ["{}/grid_median_{}_{}_{}.csv".format(params["log_folder"], i,n, tau) for i in range(10)]
    for i,file in enumerate(files):
        df = pd.read_csv(file)
        df["title"]="GPU median n={} tau={}({})".format(n,tau,i)
        df_all = df_all.append(df)

    df = pd.read_csv("{}/grid_median_all_{}_{}.csv".format(params["log_folder"], n, tau))
    df["title"]="GPU median n={} tau={}({})".format(n,tau,"all")
    df_all = df_all.append(df)

    df_all = df_all.reset_index()
    df_all.to_csv("../res_figure.csv")
    return df_all

def run_nets(params, g_stream):
    mtxs = []
    for i in range(8,10):
        print("Network {}".format(i))
        params["random_seed"]=i

        inp_args["--p_nonlin"]=get_ps(p_c=p_c, tau=params["tau"], n=params["n"], N=params["N_ps"])

        inp_args["--per"]=False
        arr,sims_cen = feed_edge_list(g_stream, inp_args, False)

        inp_args["--per"]=True
        arr,sims_per = feed_edge_list(g_stream, inp_args)

        ps = sorted(list(set(np.array(list(sims_per.keys()))[:,0])))
        ss =  [int(s) for s in inp_args["--s_nonlin"].split(' ')[1:]]

        mtx_rat = [[np.median(sims_cen[(p,s)])/np.median(sims_per[(p,s)]) for s in ss] for p in ps]
        mtx_cen = [[json.dumps(sims_cen[(p,s)].tolist()) for s in ss] for p in ps]
        mtx_per = [[json.dumps(sims_per[(p,s)].tolist()) for s in ss] for p in ps]
        save_mtx(mtx_rat, mtx_cen, mtx_per, ps, ss, params["n"], params["tau"],
                 folder=params["log_folder"], p_c = p_c, extra="median_{}".format(i))

        mtxs.append(mtx_rat)
        
    mtx_mean = np.mean(np.array(mtxs), axis=0)
    save_mtx(mtx_mean, mtx_mean, mtx_mean, ps, ss, params["n"], params["tau"],
                     folder=params["log_folder"], p_c = p_c, extra="median_all")

def run_nets(params):
    mtxs = []
    for i in range(8,10):
        print("Network {}".format(i))
        params["random_seed"]=i

        g_stream,p_c = load_graph(pop_size = params["n"],
                              deg_exp=params["tau"],
                              random_seed = i,
                              cen_type="cen",
                              folder=params["log_folder"])
        inp_args["--p_nonlin"]=get_ps(p_c=p_c, tau=params["tau"], n=params["n"], N=params["N_ps"])

        inp_args["--per"]=False
        arr,sims_cen = feed_edge_list(g_stream, inp_args, False)

        inp_args["--per"]=True
        arr,sims_per = feed_edge_list(g_stream, inp_args)

        ps = sorted(list(set(np.array(list(sims_per.keys()))[:,0])))
        ss =  [int(s) for s in inp_args["--s_nonlin"].split(' ')[1:]]

        mtx_rat = [[np.median(sims_cen[(p,s)])/np.median(sims_per[(p,s)]) for s in ss] for p in ps]
        mtx_cen = [[json.dumps(sims_cen[(p,s)].tolist()) for s in ss] for p in ps]
        mtx_per = [[json.dumps(sims_per[(p,s)].tolist()) for s in ss] for p in ps]
        save_mtx(mtx_rat, mtx_cen, mtx_per, ps, ss, params["n"], params["tau"],
                 folder=params["log_folder"], p_c = p_c, extra="median_{}".format(i))

        mtxs.append(mtx_rat)
        
    mtx_mean = np.mean(np.array(mtxs), axis=0)
    save_mtx(mtx_mean, mtx_mean, mtx_mean, ps, ss, params["n"], params["tau"],
                     folder=params["log_folder"], p_c = p_c, extra="median_all")
    
if(__name__ == "__main__"):
    # === Params ===
    params = ger_args()

    inp_args = {
        "--s_nonlin": get_ss(params["N_ss"], params["n"]),
        "--seed":0,
        "--sim_num":params["sim_num"],
        "--mode":"simulation"
    }

    # === Init networks ===
    init_graphs_parallel(pop_size = params["n"],
                          deg_exp=params["tau"], folder=params["log_folder"])
    run_nets(params)