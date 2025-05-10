import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils
from torch_geometric.data import Data,Batch
from torch_geometric.nn import GCNConv,SAGEConv,SAGPooling,global_add_pool,MLP,GATConv,global_max_pool,GINConv,BatchNorm,ResGatedGraphConv
from torch_geometric.utils import softmax
device = torch.device("cuda" if torch.cuda.is_available() else "")
class MLP_GIN(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        h = F.relu(self.batch_norm(self.linears[0](x)))
        return self.linears[1](h)


class GNNmodel(nn.Module):
    '''
    input: pyg batch,x=[num_nodes,1] 取值为0或1
    output: x=[num_nodes,1]每个节点一个值,图层面的输出graph_leval_output

    '''
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=4,
                 graph_level_output=0, learn_eps=False, dropout=0.,
                 aggregator_type="sum",graph_pool='add',conv='Gin'):
        super().__init__()

        self.inp_embedding = nn.Embedding(input_dim, #不能输入负数
                                          hidden_dim)
        # nn.embedding在于构建一个映射，input＿dim为需要进行映射的输入的数量，hidden＿dim为映射得到的结果的维度,映射为单射
     # self.inp_transform = nn.Identity()
        self.hidden_dim = hidden_dim
        self.ginlayers = nn.ModuleList()
        # self.convlayers = nn.ModuleList()
        # if conv=='Gin':
        #     mlp = MLP_GIN(hidden_dim, hidden_dim, hidden_dim)
        #     conv=GINConv(mlp, train_eps=learn_eps)
        # elif conv=='ResGated':
        #     conv=ResGatedGraphConv(hidden_dim, hidden_dim)
        self.batch_norms = nn.ModuleList()
        # assert aggregator_type in ["sum", "mean", "max"]
        for layer in range(num_layers - 1):  # excluding the input layer
            if conv=='Gin':
                mlp = MLP_GIN(hidden_dim, hidden_dim, hidden_dim)
                self.ginlayers.append(GINConv(mlp, train_eps=learn_eps))
            elif conv=='ResGated':
                self.ginlayers.append(ResGatedGraphConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))
        self.output_dim = output_dim
        self.graph_level_output = graph_level_output
        # linear functions for graph poolings of output of each layer
        self.readout = nn.Sequential(
            nn.Linear(num_layers * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim + graph_level_output)
        )
        self.drop = nn.Dropout(dropout)
        if graph_pool=='add':
            self.pool = global_add_pool
        elif graph_pool=='max':
            self.pool =global_max_pool
        else:
            raise NotImplementedError
    def forward(self, data, reward_exp=None):
        x, edge_index = data.x, data.edge_index
        h = self.inp_embedding(x)  # 在这里，输入的每一个state都是描述整个图的状态，state中只有（０，1）2种可能，在这里为每一种可能映射一个对应的结果
        # list of hidden representation at each layer
        # h = self.inp_transform(h)
        hidden_rep = [h]
        # print(h)
        for i, layer in enumerate(self.ginlayers):
            h = layer(h, edge_index)
            h = self.batch_norms[i](h)
            h = F.elu(h)
            hidden_rep.append(h)  # list num_layer*[num_nodes,hidden_dim]
        score_over_layer = self.readout(torch.cat(hidden_rep, dim=-1))

        if self.graph_level_output > 0:
            return score_over_layer[..., :self.output_dim], \
                self.pool(score_over_layer[..., self.output_dim:],data.batch)
        else:
            return score_over_layer


class test_model():
    def __init__(self):
        self.model =GNNmodel(input_dim=3, output_dim=1, graph_level_output=1).to(device)
        self.model_flow=GNNmodel(input_dim=3, output_dim=1, graph_level_output=1).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train_model(self,*batch):

        self.model.train()
        batch,b2,data=batch
        batch, b2, data = batch.to(device), b2.to(device), data.to(device)
        result1 = self.model(batch)[0]
        result2 = self.model_flow(b2)[0]
        result3=self.model_flow(data)[0]
        loss = torch.sum(result1*result2)+torch.sum(result3)*data.num_nodes
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

if __name__ == '__main__':
    # edge_index = torch.tensor([[0, 1, 0, 2],
    #                            [1, 0, 2, 1]], dtype=torch.long)
    # x = torch.tensor([1, 1, 0], dtype=torch.long)
    #
    # data = Data(x=x, edge_index=edge_index)
    # device = torch.device("cuda" if torch.cuda.is_available() else "")
    # print(data.batch)
    # g1=nx.barabasi_albert_graph(5,2)
    # d2=torch_geometric.utils.from_networkx(g1)
    # d2.x=torch.tensor([1,1,0,0,0], dtype=torch.long)
    # print(d2)
    # observation_batch = Batch.from_data_list([data, data, d2])
    # b2 = Batch.from_data_list([d2, d2, d2])
    # newbatch=Batch.from_data_list(observation_batch.to_data_list()+b2.to_data_list())
    # print(newbatch.ptr)
    # print(observation_batch.x, observation_batch.edge_index, observation_batch.batch,observation_batch)
    model=test_model()
    # newbatch=(newbatch,b2,data)
    g1 = nx.barabasi_albert_graph(10, 2)
    d2 = torch_geometric.utils.from_networkx(g1)
    d2.x=torch.tensor([-1,0,-1,0,2,2,2,2,2,2]).long()
    print(d2.x)
    newbatch=(d2,d2,d2)
    for i in range(10):
        print("1:{}".format(torch.cuda.memory_allocated(0)))
        model.train_model(*newbatch)
        print("2:{}".format(torch.cuda.memory_allocated(0)))
    g1 = nx.barabasi_albert_graph(64, 2)
    d2 = torch_geometric.utils.from_networkx(g1)
    d2.x=torch.ones([64]).long()
    newbatch=(d2,d2,d2)
    for i in range(10):
        print("3:{}".format(torch.cuda.memory_allocated(0)))
        model.train_model(*newbatch)
        print("4:{}".format(torch.cuda.memory_allocated(0)))


