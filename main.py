import json
import os
import time

import hydra
# import igraph
from omegaconf import DictConfig

from model import GNNmodel
from utils_cython import seed_all
import networkx as nx
import numpy as np
# from tensorboardX import SummaryWriter
from env import MP_Decycler
import torch
from torch_geometric.data import Batch,Data
from torch_geometric.utils import from_networkx
from multiprocessing import Process, Pool
from utils_cython import TransitionBuffer
from data_process import get_dataloader
from algo import DetailedBalanceTransitionBuffer, DetailedBalance

import os
from datetime import datetime


# 获取当前时间并格式化为字符串
now = datetime.now().strftime("%m_%d_%H_%M")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def get_alg_buffer(cfg, device):
    assert cfg.alg in ["db", "fl"]
    buffer = TransitionBuffer(size=cfg.buffer_size)  #
    alg1 = DetailedBalanceTransitionBuffer(cfg, device)  #
    return alg1, buffer


def get_logr_scaler(cfg,epoch):
    if cfg.reward_exp is None:
        reward_exp=1
    else:
        reward_exp = float(cfg.reward_exp)

    if cfg.anneal == 'linear': #前十分之一个epoch逐步增加到目标值,后保持不变
        process_ratio = max(0., min(1, epoch/cfg.anneal_epochs))  # from 0 to 1
        reward_exp = reward_exp * process_ratio + \
                     float(1) * (1 - process_ratio)
    elif cfg.anneal == "none":
        pass
    else:
        raise NotImplementedError

    # (R/T)^beta -> (log R - log T) * beta
    def logr_scaler(r):

        return r * reward_exp

    return logr_scaler


def networkx_to_pyg(graph):
    graph=graph.to_directed()
    edges = list(graph.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(edge_index=edge_index,x=torch.zeros([graph.number_of_nodes()]))
@torch.no_grad()
def rollout(gbatch, alg, device,cfg):
    '''
    智能体与环境交互获取数据
    :param gbatch: list of networkx graph
    :param alg: model used for select action
    :return: batch(s,a,r,s_,done),batch_metric
    '''
    s = []  # list of pyg batch
    a = []  # list of numpy array
    r = []  # list of numpy array
    d = []  # list of numpy array
    env = MP_Decycler(graph_list=gbatch, cpu_use=cfg.cpu_use,env_a=cfg.env_a)
    state, reward, done = env.get_state()  # retuen state and reward done
    with Pool(processes=cfg.cpu_use) as pool:
        result = pool.map(networkx_to_pyg, gbatch)

    pyg_batch = Batch.from_data_list(result).to(device)
    pyg_batch.x = torch.from_numpy(state).long().to(device)
    # s.append(pyg_batch.__copy__().to('cpu'))
    # d.append(done)
    while not np.all(env.done):  # 所有环境都结束
        action_list = alg.sample(pyg_batch, state, env.done, rand_prob=0.0)
        action_list = action_list.detach().cpu().numpy()
        s.append(pyg_batch.__copy__().to('cpu'))
        r.append(env.reward)
        a.append(action_list)
        d.append(env.done)
        state = env.step(action_list)
        pyg_batch.x = torch.from_numpy(state).long().to(device)
    s.append(pyg_batch.__copy__().to('cpu'))
    r.append(env.reward)
    d.append(env.done)
    assert len(s) == len(a) + 1 == len(r) == len(d)
    # reward_per_graph = -torch.mean(torch.from_numpy(env.reward) / (torch.diff(pyg_batch.ptr).cpu()))
    reward_per_graph=torch.mean(torch.from_numpy(env.reward))
    # a = torch.tensor(a)
    # r = torch.from_numpy(r)
    # d = torch.from_numpy(d)
    batch = s, a, r, d
    return batch, reward_per_graph.item()


@hydra.main(config_path="model_config", config_name="main")
def train(cfg:DictConfig) :
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    device = torch.device(f"cuda:{cfg.device:d}" if torch.cuda.is_available() and cfg.device>=0 else "cpu")
    print(f"Device: {device}")
    model_save = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir+"/model"
    os.makedirs(model_save, exist_ok=True)
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    train_info_dict = {}
    alg, buffer = get_alg_buffer(cfg=cfg, device=device)
    seed_all(cfg.seed)
    train_loader, test_loader = get_dataloader(train_batch_size=cfg.rollout_batch_size, test_batch_size=cfg.test_batch_size)
    trainset_size = len(train_loader.dataset)
    print(f"Trainset size: {trainset_size}")
    print('env reward_shape:',cfg.reward_shape)
    train_reward_per_epoch = []
    eval_result = []
    for ep in range(cfg.num_epochs):
        t1 = time.time()
        train_reward_per_batch = []
        logr_scaler = get_logr_scaler(cfg, ep)
        # print(logr_scaler(1))
        for batch_idx, gbatch in enumerate(train_loader):
            t2=time.time()
            batch, reward_sum = rollout(gbatch, alg, device,cfg=cfg)
            train_reward_per_batch.append(reward_sum)
            buffer.add_batch(batch)
            batch_size = min(len(buffer), cfg.batch_size)
            for _ in range(cfg.train_steps):  # 每次迭代过程中训练次数
                if not buffer.pos:
                    break
                trainbatch = buffer.sample(batch_size=batch_size)
                _ = alg.train_step(*trainbatch,
                                            logr_scaler=logr_scaler)
                # loss_list.append(loss)
            if cfg.on_policy:
                buffer.reset()
            print('batch', batch_idx, 'reward_per_batch:', reward_sum,'time',
                  time.time()-t2)
                  #,'batch_loss',sum(loss_list[-cfg.train_steps])/cfg.train_steps)
        eval_result.append(eval(alg, test_loader, device,env_a=cfg.env_a,cpu_use=cfg.cpu_use))
        train_reward = np.mean(train_reward_per_batch)
        train_reward_per_epoch.append(train_reward)
        print('epoch', ep, 'train_reward_per_epoch:', train_reward,
              'eval_reward_per_epoch:', eval_result,
              'time_used:', time.time() - t1,'reward_exp',logr_scaler(1))
        alg.save(model_save +'/' + str(ep) + "alg.pt")
    train_info_dict['train_reward_per_epoch:'] = train_reward_per_epoch
    train_info_dict['eval_reward_per_epoch:'] = eval_result
    output = os.path.join(model_save, 'train_result.json')
    with open(output, 'w') as json_file:
        json.dump(train_info_dict, json_file, indent=2)
    return 1


@torch.no_grad()
def eval(alg,testloder,device,env_a,cpu_use): #
    reward_sum=0
    for batch_idx, gbatch in enumerate(testloder):
        env = MP_Decycler(graph_list=gbatch, cpu_use=cpu_use,env_a=env_a)
        state, _,_ = env.get_state()  # retuen state and reward done
        with Pool(processes=cpu_use) as pool:
            result = pool.map(networkx_to_pyg, gbatch)
        pyg_batch = Batch.from_data_list(result).to(device)
        pyg_batch.x = torch.from_numpy(state).long().to(device)
        while not np.all(env.done):
            action_list = alg.sample(pyg_batch, _, env.done, rand_prob=0.0)
            action_list = action_list.detach().cpu().numpy()
            state = env.step(action_list)
            pyg_batch.x = torch.from_numpy(state).long().to(device)
        # reward_per_graph = -torch.mean(torch.from_numpy(env.reward) / (torch.diff(pyg_batch.ptr).cpu())).item()
        reward_per_graph = torch.mean(torch.from_numpy(env.reward)).item()
        reward_sum +=reward_per_graph
    return  reward_sum




if __name__ == '__main__':
    
    train()


