import numpy as np
import networkx as nx

from subprocess import Popen, STDOUT, PIPE
import os


def run_ecl_cc(graph, init_seeds):
    n = len(graph.nodes)
    m = 2*len(graph.edges)
    stream = "{} {} {}\n".format(n, m, len(init_seeds))

    nindex = np.zeros(n+1, dtype=np.int)
    nlist = np.zeros(m, dtype=np.int)
    eweight = np.zeros(m, dtype=np.float)
    
    nindex[0] = 0
    new_index = {n:i for i,n in enumerate(graph.nodes)}
    edges = sorted([(new_index[u], new_index[v], float(graph[u][v]["weight"])) \
            for u,v in graph.edges]+[(new_index[v], new_index[u], float(graph[u][v]["weight"])) \
            for u,v in graph.edges])
    
    for i,(src,dst,wei) in enumerate(edges):
        nindex[src+1] = i+1
        nlist[i] = dst
        eweight[i] = wei
        
    for i in range(1,n+1):
        nindex[i] = max(nindex[i-1],nindex[i])
        
    stream+=" ".join([str(i) for i in nindex])+"\n"
    stream+=" ".join([str(n) for n in nlist])+"\n"
    stream+=" ".join([str(e) for e in eweight])+"\n"
    stream+=" ".join(str(s) for s in init_seeds)+'\n'
    
    print("Graph collected")
    p = Popen([os.path.dirname(os.path.abspath(__file__))+'/../src/bin/ecl-cc', 'nofile'],
          stdout=PIPE, stdin=PIPE, stderr=STDOUT, bufsize=1, universal_newlines=True)

    out,err = p.communicate(stream)
    for line in out.split('\n')[:-1]:
        print(line)
        
    return stream

def get_graphstream(graph, init_seeds, args):
    n = len(graph.nodes)
    #m = 2*len(graph.edges)
    m = len(graph.edges)
    stream = "{} {} {}\n".format(n, m, len(init_seeds))

    new_index = {n:i for i,n in enumerate(graph.nodes)}
    #edges = [(new_index[u], new_index[v], float(graph[u][v]["weight"])) \
    #        for u,v in graph.edges]+[(new_index[v], new_index[u], float(graph[u][v]["weight"])) \
    #        for u,v in graph.edges]
    
    edges = [(new_index[u], new_index[v], float(graph[u][v]["weight"])) \
                for u,v in graph.edges]
    stream+=" ".join("{} {} {}".format(u,v,w) for u,v,w in edges)+"\n"
    stream+=" ".join(str(new_index[s]) for s in init_seeds)+'\n'
    
    if(("verbose" in args) and args["verbose"]):
        print("Graph collected")
    
    return stream

def feed_edge_list(graph_stream, args, verbose = False):
    str_args = " ".join([str(item) for pair in args.items() for item in pair])
    str_args = str_args.split(' ')
    print(str_args)
    p = Popen([os.path.dirname(os.path.abspath(__file__))+'/../src/bin/ecl-cc', 'nofile']+str_args,
          stdout=PIPE, stdin=PIPE, stderr=STDOUT, bufsize=1, universal_newlines=True)
    
    if(verbose):
        print(" ".join([os.path.dirname(os.path.abspath(__file__))+'/bin/ecl-cc', 'nofile']+str_args))
        print(graph_stream)
    
    out,err = p.communicate(graph_stream)
    
    arr = []
    sims=[]
    do_print = True
    for line in out.split('\n')[:-1]:
        if(line[:5]=="[RES]"):
            line = line.replace('\n', '')
            line = line[6:-1] if line[-1]==' ' else line[6:]
            arr= [int(a) for a in line.split(' ')]
        elif(line[:6]=="[SIMS]"):
            do_print =  False
            data = out.split("[SIMS]\n")[1]
            #data = [[float(a) for a in row[:-1].split(' ')] for row in data.split("\n")[:-1]]
            data={tuple(row.split(':')[0].split(',')):row.split(':')[1].split(' ') for row in data.split("\n")[:-1]}
            data = {(float(k[0]),int(k[1])): np.array([float(a) for a in v]) for k,v in data.items()}
            sims = data
        elif(do_print):
            print(line)
    
    print("[Python] Error: {}".format(err))
    return np.array(arr),sims