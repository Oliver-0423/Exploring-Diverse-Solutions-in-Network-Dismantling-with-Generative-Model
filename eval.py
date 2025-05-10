import os
import random
from utils import seed_all
import networkx as nx
import torch
import time
import numpy as np
import hydra
from torch_geometric.data import Data
from algo import DetailedBalanceTransitionBuffer, DetailedBalance
from env import Graph_Decycler
def networkx_to_pyg(graph):
    graph=graph.to_directed()
    edges = list(graph.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(edge_index=edge_index,x=torch.zeros([graph.number_of_nodes()]))
@torch.no_grad()
def get_sol(alg,graph,step_ratio,device,rand=False,flow=False):
    alg.model.eval()
    alg.model_flow.eval()
    pyg_graph=networkx_to_pyg(graph).to(device)
    env=Graph_Decycler(graph)
    state=env.get_graph_state()
    pyg_graph.x = torch.from_numpy(state).long().to(device)
    done=False
    action_list=[]
    flow_list=[]
    # nodes_last=[]
    t1=time.time()
    while not done:
        if rand:
            valid_action = env.valid_action()
            action=[random.choice(valid_action).item()]
            # action = [valid_action[0].item()]
        else:
            valid_action = env.valid_action()
            pf=alg.model(pyg_graph).reshape(-1).double()
            pf = pf[valid_action].softmax(dim=0)
            step=max(int(step_ratio*env.num_actions()),1)
            action=valid_action[torch.multinomial(pf,num_samples=step).cpu()].tolist()
            # action=valid_action[torch.topk(pf,step,largest=True)[1].cpu()].tolist()
            # print('step',step,'action',action)
        if flow:
            f=alg.model_flow(pyg_graph)[1].item()+env.get_reward()*500
            flow_list.append(f)
        # nodes_last.append(env.num_actions())
        state,_,done = env.step(action)
        if type(action)==int:
            action=[action]
        action_list.extend(action)
        pyg_graph.x=torch.from_numpy(state).long().to(device)
    t2=time.time()
    result = {
        'decycle_list': action_list,
        'time': t2-t1,
        'score':len(action_list)/nx.number_of_nodes(graph)
    }
    return result

@torch.no_grad()
def get_k_graph_sol(alg,graph,step_ratio,device,rand=False,flow=False):
    factor_nodes = set(node for node, attr in graph.nodes(data=True) if attr.get('node_type') == 'factor')
    alg.model.eval()
    alg.model_flow.eval()
    pyg_graph=networkx_to_pyg(graph).to(device)
    env=Graph_Decycler(graph)
    state=env.get_graph_state()
    pyg_graph.x = torch.from_numpy(state).long().to(device)
    done=False
    action_list=[]
    flow_list=[]
    # nodes_last=[]
    t1=time.time()
    while not done:
        if rand:
            valid_action = env.valid_action()
            action=[random.choice(valid_action).item()]
            # action = [valid_action[0].item()]
        else:
            valid_action = list(set(env.valid_action())-factor_nodes)
            pf=alg.model(pyg_graph).reshape(-1).double()
            pf = pf[valid_action].softmax(dim=0)
            step=max(int(step_ratio*env.num_actions()),1)
            action=valid_action[torch.multinomial(pf,num_samples=step).cpu()].tolist()
            # action=valid_action[torch.topk(pf,step,largest=True)[1].cpu()].tolist()
            # print('step',step,'action',action)
        if flow:
            f=alg.model_flow(pyg_graph)[1].item()+env.get_reward()*500
            flow_list.append(f)
        # nodes_last.append(env.num_actions())
        state,_,done = env.step(action)
        if type(action)==int:
            action=[action]
        action_list.extend(action)
        pyg_graph.x=torch.from_numpy(state).long().to(device)
    t2=time.time()
    result = {
        'decycle_list': action_list,
        'time': t2-t1,
        'score':len(action_list)/(nx.number_of_nodes(graph)-len(factor_nodes))
    }
    return result

def rand(graph):
    env = Graph_Decycler(graph)
    done = False
    action_list=[]
    t1 = time.time()
    while not done:
        valid_action = env.valid_action()
        action = [random.choice(valid_action).item()]
        state, reward, done = env.step(action)
        if type(action) == int:
            action = [action]
        action_list.extend(action)
    t2 = time.time()
    result = {
        'reward':reward,
        'decycle_list': action_list,
        'time': t2 - t1,
        'score': len(action_list) / nx.number_of_nodes(graph)
    }
    return result

def rand_eval(input_folder):
    resultdict={}
    num_graph=0
    score=0
    reward=0
    for filename in os.listdir(input_folder):
        print(filename)
        file_path = os.path.join(input_folder, filename)
        G = nx.read_edgelist(file_path, nodetype=int)
        result=rand(G)
        resultdict[filename] = result
        score = score + result['score']
        reward=reward+result['reward']
        num_graph += 1
    print('score:', score / num_graph,'reward:',reward/num_graph)
    with open('rand_test_nx_decycle.json', 'w') as json_file:
        json.dump(resultdict, json_file, indent=2)

@hydra.main(config_path="multirun//2024-07-17//14-42-56//0//.hydra", config_name="config")
@torch.no_grad()
def final_eval(cfg):
    step=0.01 #0.001:0.1026,0.01:0.10344774229250535 0.1:0.1422
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    alg = DetailedBalance(cfg, device)
    # for a in range(76,100):
    glist=[]
    name_list=[]
    score_dict={}
    input_folder = '/home/neu/sunxiaojie/gfnco2/test_data_gflow'
    # a=['corruption.graphml', 'eu-powergrid.graphml', 'foodweb-baydry.graphml', 'foodweb-baywet.graphml', 'inf-USAir97.graphml', 'maayan-foodweb.graphml',
    #    'moreno_crime_projected.graphml', 'oregon2_010526.graphml', 'econ-wm1.graphml', 'arenas-meta.graphml', 'loc-brightkite.graphml', 'maayan-Stelzl.graphml',
    #    'maayan-vidal.graphml', 'petster-hamster.graphml', 'power-eris1176.graphml', 'subelj_jung-j_jung-j.graphml', 'web-EPA.graphml', 'web-webbase-2001.graphml']
    # a.remove('k3_time')
    for filename in os.listdir(input_folder):
        print(filename)
        name_list.append(filename)
        file_path = os.path.join(input_folder, filename)
        # glist.append(igraph.Graph.Read_GraphML(file_path))
        # glist.append(nx.read_edgelist(file_path, nodetype=int))
        glist.append(nx.read_graphml(file_path, node_type=int))
    #     break
    path_list=[
        "/home/neu/sunxiaojie/gfnco2/multirun/2024-07-17/14-42-56/1/model/62alg.pt",
        # "/home/neu/sunxiaojie/gfnco2/multirun/2024-07-17/14-41-42/0/model/51alg.pt",
        # "/home/neu/sunxiaojie/gfnco2/multirun/2024-07-17/14-42-26/0/model/25alg.pt",
        # "/home/neu/sunxiaojie/gfnco2/multirun/2024-07-17/14-42-09/0/model/62alg.pt",
        # "/home/neu/sunxiaojie/gfnco2/multirun/2024-07-17/14-39-59/0/model/80alg.pt",
        # "/home/neu/sunxiaojie/gfnco2/multirun/2024-07-17/14-42-39/0/model/32alg.pt",
        # "/home/neu/sunxiaojie/gfnco2/multirun/2024-07-17/14-41-32/0/model/57alg.pt",
        # "/home/neu/sunxiaojie/gfnco2/multirun/2024-07-17/14-41-55/0/model/30alg.pt",
        # "/home/neu/sunxiaojie/gfnco2/multirun/2024-07-17/14-42-45/0/model/95alg.pt",
                ]
    for path in path_list:
        alg.load(path)
        # torch.save(alg.model,'gflow_pf.model')
        # torch.save(alg.model_flow, 'gflow_flow.model')
        # return 0
        for seed in [9]:
            seed_all(seed)
            file_path=path[48:56]+'seed_'+str(seed)+'_step'+str(step)+'test_real_decycle_model'+path[-8:-6]+'.json'
            print('current model:', path, 'file:',
                  file_path)
            score=0
            resultdict={}
            resultdict['model_file']=file_path
            # decycle_sol = {'decycle_list':{},'time_used':{},'score':{}}
            num_graph=0
            for i in range(len(glist)):
                G=glist[i]
                filename=name_list[i]
                result=get_sol(alg,graph=G,step_ratio=step,device=device,rand=False,flow=False)
                resultdict[filename] = result
                score = score + result['score']
                num_graph += 1
            print('seed:',seed,'score:',score / num_graph)
            score_dict[str(seed)+path]=score / num_graph
            with open(file_path, 'w') as json_file:
                json.dump(resultdict, json_file, indent=2)
    with open('total_score_dict.json','w') as json_file:
        json.dump(score_dict,json_file,indent=2)






if __name__ == '__main__':
    # final_eval()

    final_eval()

