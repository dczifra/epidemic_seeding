#!/usr/bin/env python
from threading import Lock
from flask import Flask, render_template, session, request, \
    copy_current_request_context
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
import os
import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from utils.graph_generator import get_graph
import argparse
from seed_research import init_graph

from country import Country
from agent_model import Agent_Country
from utils.dotdict import dotdict

def parse_args():
    parser = argparse.ArgumentParser(description='Parser for application')
    parser.add_argument('--port', dest='port', type = int, default=5000,
                    help='Choose the port for the localhost port')
    
    return parser.parse_args()

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode, ping_interval=3000, ping_timeout=6000)
thread = None
thread_lock = Lock()
boardList = {}
graphList = {}
max_iteration= 100

def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        socketio.sleep(10)
        count += 1
        socketio.emit('my_response',
                      {'data': 'Server generated event', 'count': count},
                      namespace='/test')


@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)


@socketio.on('my_event', namespace='/test')
def test_message(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})

@socketio.on('my_ping', namespace='/test')
def ping_pong():
    emit('my_pong')

@socketio.on('connect', namespace='/test')
def test_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    emit('my_response', {'data': 'Connected', 'count': 0})
    #test_signIn(testParam)
    
def start_game_continuous():
    #self.log_csv()
    while (boardList[request.sid].iterations<boardList[request.sid].args.max_iteration) and keep_going[request.sid]:
        emit("sim_response", {'result': boardList[request.sid].step()})
#        socketio.sleep(boardList[request.sid].args.sleep)
        #self.log_csv()   
        
@socketio.on('load_graph', namespace='/test')
def test_sim(message):
    print("Load")
    args = dotdict(message)
    if args.graph_type=="mask":
        args.graph_type="linklist"
        args.graph_args = {"nodes":"data/HU_id_place_pop_lat_lon.csv","edges":"data/HU_orig_dest_Freq_2020-03-23-wd-2020-05-19.csv"}
        graphList[request.sid] = get_graph(args.graph_type, args.graph_args)
    elif args.graph_type[0]=="K":
        args["graph"]="KSH"
        args["node_factor"]=25
        args["edge_factor"]=25
        args["equal_pop"] = True
        args["equal_edge"] = True
        args["shuffle_edges"] = "shuffle_edges" in args.graph_type
        graphList[request.sid] = init_graph(args)
    else:
        graphList[request.sid] = get_graph(args.graph_type, args.graph_args)

    if (args.graph_type=="mask") or (args.graph_type=="megye"): #we subsample
        population = nx.get_node_attributes(graphList[request.sid], "population")
        population = {k: v//50 for k,v in population.items()}
        nx.set_node_attributes(graphList[request.sid], population, "population")
    if (args.graph_type=="megye"):
        with open('public_data/megye.data') as fp:
            emit("graph_response", {'graph': json_graph.node_link_data(graphList[request.sid])})
            nx.set_node_attributes(graphList[request.sid], {n:0 for n in graphList[request.sid].nodes()}, 'init_node')
            emit("sim_response", {'result': list(map(json.loads,fp.readlines())),'graph': json_graph.node_link_data(graphList[request.sid])})
    else:
        emit("graph_response", {'graph': json_graph.node_link_data(graphList[request.sid])})


@socketio.on('simulate', namespace='/test')
def test_sim(message):
    print("Simulate")
    graph = graphList[request.sid]
    args = dotdict(message)

    if args["model_type"]=="metapop":
        population = nx.get_node_attributes(graph, "population")

        betas = {g: args["beta"] for g in graph.nodes()}
        if args["sigma"]==0:
            e_to_is = {g: (lambda max_bin, samples: np.zeros(samples)) for g in graph.nodes()}
        else:
            e_to_is = {g: (lambda max_bin, samples: np.clip(np.random.geometric(args["sigma"],samples), 0, max_bin)) for g in graph.nodes()}
            #e_to_is = {g: (lambda max_bin, samples: np.random.binomial(max_bin, args["sigma"],samples) ) for g in graph.nodes()}
        i_to_rs = {g: (lambda max_bin, samples: np.clip(np.random.geometric(args["gamma"],samples), 0, max_bin)) for g in graph.nodes()}

        boardList[request.sid]  = Country(graph, population, betas, e_to_is, i_to_rs, args)
    elif args["model_type"]=="nodebased":
        args["super_infected_agents"]=[]
        if len(args["init_nodes"])==0:
            args["init_nodes"]={'0': 2}
        args["infected_agents"]=list(map(lambda x: int(x), args["init_nodes"].keys()))
        boardList[request.sid]  = Agent_Country(args, graph)
    else:
        print("Model not implemented")
        return

    run_entire_game()

def run_entire_game():
    last_message=1
    plots=[]
    for i in range(boardList[request.sid].args.max_iteration):
        plots.append(boardList[request.sid].get_json())
        if boardList[request.sid].check_stop():
            print("Infection extinction at iteration: {}".format(i))
            break

        boardList[request.sid].step()
        boardList[request.sid].log_json()
        if (boardList[request.sid].args.max_iteration//10)*last_message<i:
            emit("waitbox_response", str(last_message)+"0%")
            last_message+=1
            socketio.sleep(0)
    
    emit("sim_response", {'result': plots,'graph': json_graph.node_link_data(boardList[request.sid].graph)})

@socketio.on('compAw', namespace='/test')
def comp_awareness(message):
    if boardList[request.sid].args["model_type"]=="nodebased":
        plots = boardList[request.sid].get_awareness_info(message["data"],dotdict(message["parameters"]))
    else:
        plots = message["data"]
    emit("sim_response", {'result': plots,'graph': json_graph.node_link_data(boardList[request.sid].graph)})
    
@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    if request.sid in boardList:
        del boardList[request.sid]
    print('Client disconnected', request.sid)
    print(len(boardList), ' clients left')

if __name__ == '__main__':
    args = parse_args()
    port = int(os.environ.get('PORT', args.port))
    socketio.run(app, debug=True, port= port)
