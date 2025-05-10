import random
import networkx as nx
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from utils import pad_batch
from model import GNNmodel


def sample_from_logits(pf_logits, pyg_batch, state, done, rand_prob=0.):
    pf_logits = torch.where(pyg_batch.x!=1 , float('-inf'), pf_logits)  # 屏蔽掉每一个图中不合法的动作
    pf_logits = pad_batch(pf_logits, torch.diff(pyg_batch.ptr).tolist(), padding_value=-np.inf)
    action = torch.full([pyg_batch.num_graphs, ], -1, dtype=torch.long, device=pf_logits.device)
    pf_undone = pf_logits[~done].softmax(dim=1)
    action[~done] = torch.multinomial(pf_undone, num_samples=1).squeeze(-1)  # 从pf中采样得到一个样本
    if rand_prob > 0.:
        unif_pf_undone = torch.isfinite(pf_logits[~done]).float()
        rand_action_unodone = torch.multinomial(unif_pf_undone, num_samples=1).squeeze(-1)
        rand_mask = torch.rand_like(rand_action_unodone.float()) < rand_prob
        action[~done][rand_mask] = rand_action_unodone[rand_mask]
    return action


class DetailedBalance(object):
    def __init__(self, cfg,device):
        self.device = device
        self.model = GNNmodel(3, 1, graph_level_output=0,num_layers=cfg.model.num_layer,
                              hidden_dim=cfg.model.hidden_dim,graph_pool=cfg.model.pool,learn_eps=cfg.model.learn_eps ,conv=cfg.model.conv).to(device)
        self.model_flow = GNNmodel(3, 0, graph_level_output=1, num_layers=cfg.model.num_layer,
                                   hidden_dim=cfg.model.hidden_dim,graph_pool=cfg.model.pool,learn_eps=cfg.model.learn_eps,conv=cfg.model.conv).to(device)
        self.params = [
            {"params": self.model.parameters(), "lr": cfg.model.lr},
            {"params": self.model_flow.parameters(), "lr": cfg.model.lr},
        ]
        self.optimizer = torch.optim.Adam(self.params)
        self.leaf_coef = cfg.leaf_coef

    def parameters(self):
        return list(self.model.parameters()) + list(self.model_flow.parameters())

    @torch.no_grad()
    def sample(self, pyg_batch, state, done, rand_prob=0., temperature=1., reward_exp=None):
        self.model.eval()
        pf_logits = self.model(pyg_batch)[..., 0]  # 前向采样Ｐf
        return sample_from_logits(pf_logits / temperature, pyg_batch, state, done, rand_prob=rand_prob)

    def save(self, path):
        save_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_dict.update({"model_flow": self.model_flow.state_dict()})
        torch.save(save_dict, path)
        print(f"Saved to {path}")

    def load(self, path):
        save_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(save_dict["model"])
        self.model_flow.load_state_dict(save_dict["model_flow"])
        self.optimizer.load_state_dict(save_dict["optimizer"])
        print(f"Loaded from {path}")

    def train_step(self, *batch):
        raise NotImplementedError


class DetailedBalanceTransitionBuffer(DetailedBalance):
    def __init__(self, cfg, device):
        assert cfg.alg in ["db", "fl"]
        self.forward_looking = (cfg.alg == "fl")
        super(DetailedBalanceTransitionBuffer, self).__init__( cfg,device)

    def train_step(self, *batch,logr_scaler):
        self.model.train()
        self.model_flow.train()
        torch.cuda.empty_cache()

        s, logr, a, s_next, logr_next, d = batch
        s, logr, s_next, logr_next, d=(s.to(self.device), logr.to(self.device),
                                           s_next.to(self.device), logr_next.to(self.device), d.to(self.device))

        logr, logr_next = logr_scaler(logr), logr_scaler(
            logr_next)  # 定义的reward就是 选中节点个数的exponential，所以我们的log_reward就是选中的节点个数,这里再乘温度系数。
        batch_size = s.num_graphs
        flows=self.model_flow(s)[1].reshape(-1)
        flows_next=self.model_flow(s_next)[1].reshape(-1)
        pf_logits = self.model(s).reshape(-1) # 对应公式中的
        pf_logits[s.x!=1]=-np.inf
         # 把已经确定状态的节点(不为1的节点)设为-inf,因为他们不可能被前向选择
        pf_logits = pad_batch(pf_logits, torch.diff(s.ptr).tolist(),
                              padding_value=-np.inf)  # 将一维的节点向量变成矩阵,大小为[batch_size*num_nodes_max],用-inf填充,用于下一步的log_softmax
        log_pf = F.log_softmax(pf_logits, dim=1)[
            torch.arange(batch_size), a]  # 从处理过的结果中选择对应图的对应动作节点所对定的概率.最终得到一个batch_size大小的tensor
        #这里B没有利用神经网络，而是设置成一个uniform distribution（以相同概率选择一个节点remove），所以概率的值就是 1 / 父节点个数
        log_pb = torch.tensor(
            [torch.log(1 / (s_==2).sum())  # s_=s',计算每一个s_next到s的概率,最后得到一个batch_size大小的tensor
             # 获取父节点的数目求和取倒数，状态中0的数量就是所有可能的父节点的数量
             for s_ in torch.split(s_next.x, torch.diff(s.ptr).tolist(), dim=0)]).to(self.device)
        # 由cfg．ａｌｇ决定的核心差异，fl为前者
        if self.forward_looking:
            flows_next[d]= 0  # \tilde F(x) = F(x) / R(x) = 1, log 1 = 0
            lhs = logr + flows + log_pf  # (bs,)
            rhs = logr_next + flows_next + log_pb  # logr-logr_next=E(s-s')
            loss = (lhs - rhs).pow(2)
            loss = loss.mean()
        else:  # 这里为简单的detailed balance
            flows_next = torch.where(d, logr_next, flows_next)  # 如果s'是最终节点的话，用最终得到的奖励值来代替模型估计的流量
            lhs = flows + log_pf  # (bs,)
            rhs = flows_next + log_pb
            losses = (lhs - rhs).pow(2)
            loss = (losses[d].sum() * self.leaf_coef + losses[~d].sum()) / batch_size  # 给终止节点更多的权重，leaf_coef为给终止节点上加的权重
            # del losses
        # return_dict = {"train/loss": loss.item()}
        #print("11:{}".format(torch.cuda.memory_allocated(0)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


if __name__ == "__main__":
    from model import GIN
    from torch_geometric.data import Data, Batch
    import torch_geometric
    import torch
    import numpy as np
    import networkx as nx

    from model import GIN
    from algo import sample_from_logits

    device = torch.device("cuda" if torch.cuda.is_available() else "")
    edge_index = torch.tensor([[0, 1, 0, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([1, 1, 0], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)

    # #print(data.batch)
    g1 = nx.barabasi_albert_graph(5, 2)
    d2 = torch_geometric.utils.from_networkx(g1)
    d2.x = torch.tensor([1, 1, 0, 0, 0], dtype=torch.long)
    # #print(d2)
    observation_batch = Batch.from_data_list([data, data, d2])
    b2 = Batch.from_data_list([d2, d2, d2])
    newbatch = Batch.from_data_list(observation_batch.to_data_list() + b2.to_data_list())
    #print(newbatch.ptr)
    #print(newbatch.x, newbatch.edge_index, newbatch.batch, newbatch)
    model = GIN(input_dim=2, output_dim=1, graph_level_output=1)
    model = model.to(device)
    newbatch = newbatch.to(device)
    output = model(newbatch)
    pred = output[0]
    #print(pred)
    done = np.array([True, False, False, True, False, False])
    #print(sample_from_logits(pf_logits=pred, pyg_batch=newbatch, state=0, done=done))
