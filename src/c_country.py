from subprocess import Popen, STDOUT, PIPE

import os
import sys
import copy
import numpy as np
import networkx as nx
import multiprocessing
from utils.dotdict import dotdict
from utils.graph_generator import get_graph
import matplotlib.pyplot as plt

global shm

def iter_by2(iterable):
    return zip(*[iter(iterable)]*2)

def get_Chi(aggregated, cities):
    ratioInf=aggregated[:,2]/(np.sum(aggregated, axis = 1))
    
    xi = cities[:,:,2]
    mi = np.sum(cities,axis=2)
    
    sumChi2 = np.sum((xi-mi)**2/mi, axis=1)
    
    return sumChi2

class C_Country:
    def __init__(self, graph):
        # Parameters:
        #     graph      : network of cities
        self.graph = graph
        self.init_cities()

    def init_cities(self):
        # === Init Agents ===
        population = nx.get_node_attributes(self.graph, "population")
        self.agent_num = np.sum(list(population.values()))
        self.city_num = len(self.graph.nodes)

        indexes = nx.get_node_attributes(self.graph, "index")
        for i,ind in enumerate(indexes):
            if i!= ind: assert(1)

    def get_str_graph(self):
        # === City data ===
        str_graph = "{} {}\n".format(self.agent_num, self.city_num)
        for city in range(len(self.graph.nodes())):
            weights = [str(self.graph[city][neigh]["weight"]) for neigh in self.graph[city].keys()]
            neighs = [str(k) for k in self.graph[city].keys()]
            pop = self.graph.nodes[city]["population"]

            line = "{} {} {} {}\n".format(pop, len(neighs), " ".join(neighs), " ".join(weights))
            str_graph += line
        return str_graph

    @staticmethod
    def infect_cities(mode, area, inf_agent_num, pop = None):
        # TODO: use pandas
        if(mode == "uniform_random"):
            agent_location = np.random.choice(area, size = inf_agent_num)
        elif(mode == "pop_proportioned"):
            assert(pop != None)
            pop_prob = np.array(pop)/np.sum(pop)
            agent_location = np.random.choice(area, p=pop_prob, size = inf_agent_num)
        else:
            print("Mode not implemented yet")
        
        uniq, count = np.unique(agent_location, return_counts = True)
        return uniq, count
    
    @staticmethod
    def set_shm(job_num):
        job_count = multiprocessing.Value("i", 0)
        global shm
        shm = (job_count, job_sum)
    
    @staticmethod
    def run(args, str_graph, city_num, job_count, lock, inf_cities, inf_agents, verbose=False):
        # === Infect ===
        # === Inf cities ===
        str_inf_agents = "{}\n".format(len(inf_cities))
        for inf_city, agent_num in zip(inf_cities, inf_agents):
            str_inf_agents += "{} {}\n".format(inf_city, agent_num)
        
        # === Agent data ===
        # TODO: agents should be initialized in C++ 
        str_args = [str(item) for pair in args.items() for item in pair]
        p = Popen([os.path.dirname(os.path.abspath(__file__))+'/bin/main']+ str_args,
                  stdout=PIPE, stdin=PIPE, stderr=STDOUT, bufsize=1, universal_newlines=True)

        out,err = p.communicate(str_graph + str_inf_agents)

        history = []
        cities = []
        if(verbose):
            for line,line2 in iter_by2(out.split('\n')[:-1]):
                history.append([int(a) for a in line[:-1].split(" ")])
                cities.append([[int(s) for s in city.split(" ")] for city in line2[:-1].split(";")])
        else:
            for line in out.split('\n')[:-1]:
                history.append([int(a) for a in line[:-1].split(" ")])
        history = np.array(history[:])
        cities = np.array(cities)

        with lock:
            job_count[0]+=1
            print('\r {}/{}'.format(job_count[0], job_count[1]), end='', flush=True)
        return history,cities
    def run_for_betas_simple_raw(self, args, centrum, betas, inf_city_num,
                             inf_mode = "uniform_random"):
        # === INIT ===
        np.random.seed(0)
        verbose = False
        pops = nx.get_node_attributes(self.graph, "population")
        str_graph = self.get_str_graph()
        city_num = self.city_num
        agent_num = self.agent_num

        # === RUN ASYNCHRON ===
        res = {}
        pool = multiprocessing.Pool(processes=args["procnum"])
        manager = multiprocessing.Manager()
        lock = manager.Lock()

        job_count = manager.Array("i", [0,len(betas)*args["simnum"]])
        for beta in betas:
            res[beta]=[]
            for i in range(args["simnum"]):
                act_args = copy.copy(args)
                act_args["--seed"]=i
                act_args["--beta"]=beta
                inf_area = np.random.choice(centrum, size = inf_city_num, replace = False)
                inf_pop = [pops[city] for city in inf_area]
                inf_cities, inf_agents = C_Country.infect_cities(
                    inf_mode,inf_area,args["inf_agent_num"],inf_pop)
                history1 = pool.apply_async(C_Country.run, args =
                       (act_args, str_graph, city_num, job_count, lock,
                        inf_cities, inf_agents, verbose))
                #history1 = hun.run(args, inf_area)
                res[beta].append(history1)
        pool.close()
        pool.join()
        #print('') # After logging
        
        # === EVAL RESULT ===
        for beta in betas:
            res[beta] = [t1.get() for t1 in res[beta]]
        
        y = []
        for beta in betas:
            arr = [t1[0][-1,3] for t1 in res[beta]]
            y.append(arr)
        return np.array(y)
     
    def run_for_betas_simple(self, args, centrum, betas, inf_city_num,
                             inf_mode = "uniform_random"):
        # === INIT ===
        np.random.seed(0)
        verbose = True if (("verbose" in args) and args["verbose"]) else False
        pops = nx.get_node_attributes(self.graph, "population")
        str_graph = self.get_str_graph()
        city_num = self.city_num
        agent_num = self.agent_num

        # === RUN ASYNCHRON ===
        res = {}
        pool = multiprocessing.Pool(processes=args["procnum"])
        manager = multiprocessing.Manager()
        lock = manager.Lock()

        job_count = manager.Array("i", [0,len(betas)*args["simnum"]])
        for beta in betas:
            res[beta]=[]
            for i in range(args["simnum"]):
                act_args = copy.copy(args)
                act_args["--seed"]=i
                act_args["--beta"]=beta
                inf_area = np.random.choice(centrum, size = inf_city_num, replace = False)
                inf_pop = [pops[city] for city in inf_area]
                inf_cities, inf_agents = C_Country.infect_cities(
                    inf_mode,inf_area,args["inf_agent_num"],inf_pop)
                history1 = pool.apply_async(C_Country.run, args =
                       (act_args, str_graph, city_num, job_count, lock,
                        inf_cities, inf_agents, verbose))
                #history1 = hun.run(args, inf_area)
                res[beta].append(history1)
        pool.close()
        pool.join()
        #print('') # After logging
        
        # === EVAL RESULT ===
        for beta in betas:
            res[beta] = [t1.get() for t1 in res[beta]]
        
        y = []
        for beta in betas:
            arr = [t1[0][-1,3] for t1 in res[beta]]
            a = np.mean(arr, 0)
            std1 = np.std(arr, 0)
            A = 1.645/np.sqrt(len(arr)) # 90% confidence
            conf1 = (a-std1*A, a+std1*A)
            y.append((a,beta, std1, conf1[0], conf1[1]))
            
        if(verbose):
            all_chi = {}
            for beta in betas:
                chi1 =  [get_Chi(np.array(t1[0]), np.array(t1[1])) for t1 in res[beta]]
                all_chi[beta]=chi1
            return np.array(y), all_chi, res
        else:
            return np.array(y), {}
        
    def run_for_betas(self, args, centrum, betas, inf_city_num,
                      inf_mode = "uniform_random", periphery_and_centrum = False):
        # === INIT ===
        np.random.seed(0)
        pops = nx.get_node_attributes(self.graph, "population")
        if(periphery_and_centrum):
            periphery = [n for n in self.graph.nodes()]
        else:
            periphery = [n for n in self.graph.nodes() if n not in centrum]            

        str_graph = self.get_str_graph()
        city_num = self.city_num
        agent_num = self.agent_num

        # === RUN ASYNCHRON ===
        res = {}
        pool = multiprocessing.Pool(processes=args["procnum"])
        manager = multiprocessing.Manager()
        lock = manager.Lock()

        job_count = manager.Array("i", [0,2*len(betas)*args["simnum"]])
        for beta in betas:
            res[beta]=[]
            for i in range(args["simnum"]):
                act_args = copy.copy(args)
                act_args["--seed"]=i
                act_args["--beta"]=beta
                inf_area = np.random.choice(centrum, size = inf_city_num, replace = False)
                inf_pop = [pops[city] for city in inf_area]
                inf_cities, inf_agents = C_Country.infect_cities(
                    inf_mode,inf_area,args["inf_agent_num"],inf_pop)
                history1 = pool.apply_async(C_Country.run, args =
                       (act_args, str_graph, city_num, job_count, lock, inf_cities, inf_agents))
                #history1 = hun.run(args, inf_area)
                
                inf_area = np.random.choice(periphery, size = inf_city_num, replace = False)
                inf_cities, inf_agents = C_Country.infect_cities(
                    inf_mode,inf_area,args["inf_agent_num"],inf_pop)
                history2 = pool.apply_async(C_Country.run, args =
                       (act_args, str_graph, city_num, job_count, lock, inf_cities, inf_agents))
                #history2 = hun.run(args, inf_area)
                res[beta].append((history1,history2))
        pool.close()
        pool.join()
        #print('') # After logging
        
        # === EVAL RESULT ===
        for beta in betas:
            res[beta] = [(t1.get(),t2.get()) for t1,t2 in res[beta]]
        
        y = []
        for beta in betas:
            arr = [(t1[0][-1,3],t2[0][-1,3]) for t1,t2 in res[beta]]
            a,b = np.mean(arr, 0)
            std1,std2 = np.std(arr, 0)
            A = 1.645/np.sqrt(len(arr)) # 90% confidence
            conf1 = (a-std1*A, a+std1*A)
            conf2 = (b-std2*A, b+std2*A)
            conf = (conf1[0]/conf2[1], conf1[1]/conf2[0])
            y.append((a/b, beta, conf[0], conf[1], std1, std2))
            
        all_chi = {}
        for beta in betas:
            chi1 =  [get_Chi(np.array(t1[0]), np.array(t1[1])) for t1,t2 in res[beta]]
            chi2 =  [get_Chi(np.array(t2[0]), np.array(t2[1])) for t1,t2 in res[beta]]
            all_chi[beta]=(chi1,chi2)
        
        return np.array(y), all_chi
    
    def run_async(self, args, inf_area, inf_mode = "uniform_random"):
        str_graph = self.get_str_graph()
        city_num = self.city_num
        agent_num = self.agent_num
        pops = nx.get_node_attributes(self.graph, "population")
        inf_pop = [pops[city] for city in inf_area]

        # === RUN ASYNCHRON ===
        pool = multiprocessing.Pool(processes=args["procnum"])
        manager = multiprocessing.Manager()
        lock = manager.Lock()

        res = []
        job_count = manager.Array("i", [0,args["simnum"]])
        for i in range(args["simnum"]):
            act_args = copy.copy(args)
            act_args["--seed"]=i
            inf_cities, inf_agents = C_Country.infect_cities(
                    inf_mode,inf_area,args["inf_agent_num"],inf_pop)
            history = pool.apply_async(C_Country.run, args =
                                        (act_args, str_graph, city_num, job_count, lock, inf_cities, inf_agents))
            res.append(history)
        pool.close()
        pool.join()
        print('')

        return [np.array(hist.get()[0]) for hist in res],[np.array(hist.get()[1]) for hist in res]
    
    def run_area(self, args, area, inf_city_num, inf_mode = "uniform_random"):
        str_graph = self.get_str_graph()
        city_num = self.city_num
        agent_num = self.agent_num
        pops = nx.get_node_attributes(self.graph, "population")

        # === RUN ASYNCHRON ===
        pool = multiprocessing.Pool(processes=args["procnum"])
        manager = multiprocessing.Manager()
        lock = manager.Lock()

        res = []
        job_count = manager.Array("i", [0,args["simnum"]])
        for i in range(args["simnum"]):
            act_args = copy.copy(args)
            act_args["--seed"]=i
            inf_area = np.random.choice(area, size = inf_city_num, replace = False)
            inf_pop = [pops[city] for city in inf_area]
            inf_cities, inf_agents = C_Country.infect_cities(
                    inf_mode,inf_area,args["inf_agent_num"],inf_pop)
            history = pool.apply_async(C_Country.run, args =
                                        (act_args, str_graph, city_num, job_count, lock, inf_cities, inf_agents))
            res.append(history)
        pool.close()
        pool.join()
        print('')

        return [np.array(hist.get()[0]) for hist in res],[np.array(hist.get()[1]) for hist in res]

