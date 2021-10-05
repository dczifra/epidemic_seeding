import csv
import json
import numpy as np
import networkx as nx
from numba import jit

# JIT cannot handle yet with string/char arrays, that's why
#    I have to use int arrays. The states are:
S = 0
E = 1
I = 2
R = 3
G = 10


class Agent_Country:
    def __init__(self, args, graph):
        self.iterations = 0

        self.args = args
        self.graph = graph
        self.N = len(graph.nodes)
        self.pos = nx.get_node_attributes(self.graph, "pos")
        
        # === Init states ===
        self.init_states()
        
        # === Init distances ===
        if (("awR" in self.args) and self.args.awR!=-1):
            self.init_distances(graph)
        else:
            self.dist_bfs_from=np.zeros(shape = (self.N, self.N), dtype = np.float32)

        # === Infections ===
        if("infected_agents" in args):
            self.init_seeds = args["infected_agents"] + args["super_infected_agents"]
            self.seed[self.init_seeds] = self.init_seeds

            self.states[args["infected_agents"]] = I
            self.states[args["super_infected_agents"]] = G
            
            self.timers[self.init_seeds] = np.random.geometric(self.args["gamma"])+1
            
            indexes = nx.get_node_attributes(graph, "index")
            nx.set_node_attributes(self.graph,
                               {n: 1 if indexes[n] in self.init_seeds
                                else 0 for n in self.graph.nodes()}, 'init_node')
        else:
            print("Please give the infeceted, and superinfected agents (infected_agents, super_infected_agents)")

    def barabasi_inf(self, graph):
        ind = np.random.randint(0,self.N)
        inf_nodes = np.array([node for node in graph.neighbors(ind)])[:10]
        self.states[inf_nodes] = np.random.choice(np.array([I,G]), inf_nodes.size)
        self.timers[inf_nodes] = np.random.geometric(self.args["gamma"])+1

    def grid_inf(self):
        # === Infect some agents in the center ===
        # IGI
        # GIG
        # IGI
        n = int(np.sqrt(self.N))
        c = n//2
        inf_nodes = np.array([n*(c-1)+c-1, n*(c-1)+c, n*(c-1)+c+1, n*c+c-1, n*c+c, n*c+c+1, n*(c+1)+c-1, n*(c+1)+c, n*(c+1)+c+1])
        self.states[inf_nodes] = np.array([G,I,G,I,G,I,G,I,G])
        #self.states[inf_nodes] = np.array([1,10,1,10,1,10,1,10,1])
        self.timers[inf_nodes] = np.random.geometric(self.args["gamma"])+1


    def run(self):
        self.log_json()
        for i in range(self.args["max_iteration"]):
            if self.check_stop():
                print("Infection extinction at iteration: {}".format(i))
                break

            self.step()
            self.log_json()
        
        print("")
    
    def check_stop(self):
        return (np.sum(self.states == I)+ np.sum(self.states == G) == 0)

    def check_stop2(self):
        return not ( (self.seed[self.states == I]==self.init_seeds[0]).any() and (self.seed[self.states == I]==self.init_seeds[1]).any() )
    
    def step(self):
        #print("\rStep {}".format(self.iterations), end = '')

        p_super = self.args["p_super"]
        beta = self.args["beta"]
        beta_super = self.args["beta_super"]

        gamma = self.args["gamma"]
        xi = self.args["xi"]
        # === Infections ===
        merged = Agent_Country.update_neighs(self.states, self.indexes, self.timers, self.Rtimers,
                              self.neighs, self.seed, p_super, beta, beta_super, gamma, xi, self.args.awM, self.args.awR, self.dist_bfs_from)

        self.iterations += 1
        return merged

    @jit(nopython = True)
    def update_neighs(states, indexes, timers, Rtimers, neighs, seed, p_super,
                      beta, beta_super, gamma, xi, awM, awR, dist_bfs_from):
        # 0 : Suscepted
        # 1 : Infected
        # 10: Super Infected
        neigh_arr = neighs[0]
        slices = neighs[1]


        if xi<=0:
            states[timers==1]=R
        elif xi>=1:
            states[timers==1]=S
        else:
            states[timers==1]=R
            Rtimers[timers==1] =np.random.geometric(xi, size=np.sum(timers==1))
            states[Rtimers==1]=S
            Rtimers[Rtimers>0]-=1
            
        seed[timers==1] = -1
        timers[timers>0]-=1

        
        #if (maxInf>0) and (maxInf<int(np.sum(states == I)) + int(np.sum(states == G)):
        #    beta_super=beta_super/10
        #    beta=beta/10
        #    print("Max reached")
        
        # === Global awareness ====
        if ((awM>0) and (awR<=0)):
            numInf=int(np.sum(states == I)) + int(np.sum(states == G))
            beta_super=beta_super*(1-awM)**numInf
            beta=beta*(1-awM)**numInf

        merged =False
        for ind in indexes[(states == I) | (states == G)]:            
            adj_list = neigh_arr[slices[ind][0]:slices[ind][1]]
            S_adj_list = adj_list[states[adj_list]==0]
            if ((seed[adj_list[seed[adj_list]>0]]!=seed[ind]).any()):
                    merged=True

            # === Distance sensitive awareness ====
            awareness =1
            if ((awM>0) and (awR>0)):
                awereness_vector = awM*(dist_bfs_from[ind][((states == I) | (states == G)) & (indexes !=ind)]**(-awR))
                if len(awereness_vector)>0:
                    awareness = np.prod(1- awereness_vector)

            beta_super_current=beta_super*awareness
            beta_current=beta*awareness

            # === Choose infected: ===
            if(states[ind]==G):
                #infected = S_adj_list
                new_inf_num = np.random.binomial(S_adj_list.size, beta_super_current)
                infected = np.random.choice(S_adj_list, replace = False, size=new_inf_num)
            else:
                new_inf_num = np.random.binomial(S_adj_list.size, beta_current)
                infected = np.random.choice(S_adj_list, replace = False, size=new_inf_num)
            
            states[infected] = -I
            timers[infected] = np.random.geometric(gamma, size=len(infected))+1
            seed[infected] = seed[ind]
        
            # === Choose super infected ===
            new_super_inf_num = np.random.binomial(infected.size, p_super)
            focus = np.random.choice(infected, replace = False, size=new_super_inf_num)
            states[focus] = -G

        for i,s in enumerate(states):
            if s<0:
                states[i]=abs(s)
                
        return merged

    # === Helper Functions ===
    def get_neigh_flattened(self, neighs):
        arr = []
        slices = np.zeros(shape = (len(neighs),2))
        ind = 0
        for i,adj_list in enumerate(neighs):
            n = len(adj_list)
            slices[i] = (ind, ind+n)
            arr += adj_list

            ind += n

        return (np.array(arr, dtype = np.int32), slices)

    def init_states(self):
        #self.states = np.array(["S"]*self.N)
        self.states = np.zeros(self.N, dtype = np.int8)
        self.indexes = np.array(np.arange(self.N, dtype= np.int32))
        self.timers = np.zeros(self.N, dtype = np.int32)
        self.Rtimers = np.zeros(self.N, dtype = np.int32)
        self.seed = -np.ones(self.N, dtype=np.int32)
        
        all_neighs = []
        indexes = nx.get_node_attributes(self.graph, "index")
        for name in self.graph.nodes():
            neighs = [indexes[neigh] for neigh in self.graph[name].keys()]
            all_neighs.append(neighs)

        self.neighs = self.get_neigh_flattened(all_neighs)

    def init_distances2(self, graph):
        indexes = nx.get_node_attributes(graph, "index")
        pos = nx.get_node_attributes(graph, "pos")
        N = len(pos)
        paths = nx.shortest_path(graph)

        print(indexes)
        self.dist_l2_from = np.zeros(shape = (N, N), dtype = np.float32)
        self.dist_bfs_from = np.zeros(shape = (N, N), dtype = np.float32)

        for node1 in graph.nodes():
            for node2 in graph.nodes():
                a = np.array(pos[node1])
                b = np.array(pos[node2])
                self.dist_l2_from[indexes[node1]][indexes[node2]] = np.linalg.norm(a-b)
                self.dist_bfs_from[indexes[node1]][indexes[node2]] = float(len(paths[node1][node2])-1)

    @jit(nopython = True)
    def init_dist_l2(pos):
        N = len(pos)
        mtx = np.zeros(shape = (N, N), dtype = np.float32)

        for i in range(N):
            for j in range(N):
                a = pos[i]
                b = pos[j]
                mtx[i][j] = np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

        return mtx
    
    def init_distances(self, graph):
        indexes = nx.get_node_attributes(graph, "index")
        N = len(indexes)
        nodelist = [indexes[i] for i in graph.nodes()]
        pos = np.array([np.array(graph.nodes[node]["pos"], dtype = np.float32) for node in graph.nodes])
        self.dist_l2_from = Agent_Country.init_dist_l2(pos)
        self.dist_bfs_from = np.array(nx.floyd_warshall_numpy(graph,  weight = "dist").astype(float))
        
    def log_json(self):
        # === Delete previous logs ===
        if self.iterations == 0:
            with open(self.args["logfile"], 'w') as f:
                pass
        
        with open(self.args["logfile"], 'a') as outfile:
            outfile.write(json.dumps(self.get_json())+"\n")

    def get_json(self):
        act_iter_info = {"Iteration": self.iterations, "node_data":{}}

        # === Log cities ===
        for i,name in enumerate(self.graph.nodes()):
            agent_info = self.get_agent_info(i,name)
            if(i in self.init_seeds):
                agent_info["d_seed"] = self.get_seed_info(i)
                    
                dists_l2 = self.get_seed_descendent_dist(i, "l2")
                mean_dist_l2, agent_info["b_distance_l2"] = Agent_Country.encode_dists(dists_l2)
                agent_info["d_distance_l2"] = {"I": float(mean_dist_l2) }
         
                dists_bfs = self.get_seed_descendent_dist(i, "bfs")
                mean_dist_bfs, agent_info["b_distance_bfs"] = Agent_Country.encode_dists(dists_bfs)
                agent_info["d_distance_bfs"] = {"I": float(mean_dist_bfs) }
            
            act_iter_info["node_data"][str(i)] = agent_info
        
        # === Log country ===
        act_iter_info["agg_data"] = self.get_country_info()
        return act_iter_info

    def get_seed_info(self, index):
        if(index not in self.init_seeds):
            return False
        else:     
            data = {}
            data["I"] = int(np.sum((self.seed == index) & (self.states == I)))
            data["G"] = int(np.sum((self.seed == index) & (self.states == G)))
            return data


    def encode_dists(dists):
        if len(dists)==0:
            return (0,{0:0})
        hist, edges = np.histogram(dists, range= (0.0, dists.max()))
        return (np.mean(dists),{str(edges[i]):int(hist[i]) for i in range(len(hist))})

    def get_seed_descendent_dist(self, seed_ind, metric):
        descendents = (self.seed == seed_ind)
        if (self.args.awR==-1):
            return []

        if metric == "l2":
            distances = self.dist_l2_from[seed_ind, descendents]
        elif metric == "bfs":
            distances = self.dist_bfs_from[seed_ind, descendents]

        #print(distances.ndim)
        return np.array(distances).flatten()

        
    def get_agent_info(self, index, name):
        data = {}
        data["index"] = index
        data["name"] = name
        data["population"] = 1 # This is the original
        data["d_SIR"] = {
            "S": int(self.states[index] == S),
            "I": int(self.states[index] == I),
            "G": int(self.states[index] == G),
            "R": int(self.states[index] == R),
        }
        return data

    def get_country_info(self):
        data = {"d_SIR": {} }
        data["d_SIR"]["S"] = int(np.sum(self.states == S))
        data["d_SIR"]["I"] = int(np.sum(self.states == I))
        data["d_SIR"]["G"] = int(np.sum(self.states == G))
        data["d_SIR"]["R"] = int(np.sum(self.states == R))

        return data

    def get_awareness_info(self,data,args):
        for datai in range(len(data)):
            datap=data[datai]
            states=np.zeros(len(datap['node_data'].keys()))
            for node in datap['node_data'].keys():
                states[int(node)]=datap['node_data'][node]["d_SIR"]["I"] | datap['node_data'][node]["d_SIR"]["G"]
            awareness_vector=[]
            for node in datap['node_data'].keys():
                ind=int(node)
                awareness =1
                if datap['node_data'][node]["d_SIR"]["R"]:
                    awareness = 0
                elif (args.awM>0):
                    awereness_vector = args.awM*(self.dist_bfs_from[ind][(states==1) & (self.indexes !=ind)]**(-args.awR))
                    if len(awereness_vector)>0:
                        awareness = np.prod(1- awereness_vector)
                data[datai]['node_data'][node]['d_aw']={"I":awareness}
                awareness_vector.append(awareness)
            mean_aw, data[datai]['agg_data']["b_aw"] = Agent_Country.encode_dists(np.array(awareness_vector))
            data[datai]['agg_data']["d_aw"]={"I":float(mean_aw)}
        return data
