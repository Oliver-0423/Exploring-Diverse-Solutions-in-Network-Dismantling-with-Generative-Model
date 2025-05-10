import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from multiprocessing import Process, Pool
ENV_A=0
# ENV_B=1
REWARD_SHAPE=None
# print('00 reward number',ENV_A)
class Graph_Decycler:
    def __init__(self, Graph):
        self.graph = Graph
        self.reward = 0
        self.num_nodes = nx.number_of_nodes(self.graph)
        # self.degree=np.array(nx.degree(self.graph))[:,1]
        # self.action_num=0.0
        Nodes_original1 = self.graph.nodes()  # 记录原始节点
        self.g_nx_core = nx.k_core(self.graph, 2)  # 获取网络的2核
        Nodes_core1 = self.g_nx_core.nodes()  # 记录2核节点
        Nodes_removed1 = list(set(Nodes_original1) - set(Nodes_core1))  # 记录移除的节点
        self.graph_feat = np.full(self.num_nodes, fill_value=1, dtype=np.uint8)
        if len(Nodes_removed1) != 0:  # 检测是否存在需要移除的节点
            self.graph_feat[Nodes_removed1] = 0  # 将其标记为0

    def step(self, action):  # [1]
        if action == -1:
            return self.graph_feat, self.get_reward(), self.is_terminal()
        assert action in self.valid_action()  # 保证动作节点依然存在(为1)
        Nodes_original1 = self.g_nx_core.copy().nodes()  # [0123456]
        self.g_nx_core.remove_node(action)  # [023456]
        # self.action_num=self.action_num+1
        self.g_nx_core = nx.k_core(self.g_nx_core, 2)  # 获取移除上述节点后网络的2核
        Nodes_core1 = self.g_nx_core.nodes()  # 记录2核节点 #[023456]
        Nodes_removed1 = list(set(Nodes_original1) - set(Nodes_core1))  # [1]记录移除的节点
        # self.action_seq.extend(Nodes_removed1)  # [71]
        self.graph_feat[Nodes_removed1] = 0
        self.graph_feat[action] = 2
        # self.reward=self.reward-self.graph_feat.mean()
        return self.graph_feat, self.get_reward(), self.is_terminal()

    def get_graph_state(self):
        return self.graph_feat

    def is_terminal(self):
        return (self.graph_feat != 1).all()

    # def get_reward(self):
    #     return -(self.graph_feat == 2).sum()
    def get_reward(self):

        # return -(self.graph_feat == 2).sum()
        # return -self.action_num
        # print('03 reward number:', ENV_A)
        # return -nx.number_of_edges(nx.subgraph(self.graph,actions))
        # if any(actions):
        #     return -actions.size+(self.degree[actions].mean())
        # else:
        #     return 0
        if self.is_terminal():
            actions = np.where(self.graph_feat == 2)[0]
            if any(actions):
                graph1=self.graph.copy()
                graph1.remove_nodes_from(actions)
                return -(1)*len(actions)+ENV_A*(nx.number_of_edges(self.graph)-nx.number_of_edges(graph1))
        else:
           return 0
    def valid_action(self):
        return np.where(self.graph_feat == 1)[0]

    def action_mask(self):
        return self.graph_feat == 1



class MP_Decycler:  # 多线程包装的运行环境,输入一个batch的图进行处理

    def __init__(self, graph_list,cpu_use=None,env_a=0):
        global REWARD_SHAPE,ENV_A
        # REWARD_SHAPE=reward_shape
        # ENV_B = float(env_b)
        ENV_A=float(env_a)
        self.done = None
        self.reward = None
        self.state = None
        self.cpu_use = cpu_use
        self.pool = Pool(processes=cpu_use)
        self.env_list = self.pool.map(self.init_graph, graph_list)
        print('01 env reward number:', ENV_A)
        # print('current reward shape:',REWARD_SHAPE)

    @staticmethod
    def init_graph(graph):
        # print(graph)
        # 在这里创建 Graph_Decycler 类的实例并处理单个图
        return Graph_Decycler(graph)

    @staticmethod
    def get_one_state(env):
        return env.get_graph_state(), env.is_terminal(),env.get_reward()

    @staticmethod
    def step_graph(merged_list):
        # print('02 env reward number:', ENV_A)
        s, r, d = merged_list[0].step(merged_list[1])
        return s, r, d, merged_list[0]

    def step(self, action):
        assert len(action) == len(self.env_list)
        merged_list = list(zip(self.env_list, action))
        result = self.pool.map(self.step_graph, merged_list)
        s, r, d, self.env_list = zip(*result)
        self.state=np.concatenate(s)
        self.reward=np.array(r)
        self.done=np.array(d)
        return self.state

    def get_state(self):
        result = self.pool.map(self.get_one_state, self.env_list)
        state, done,reward = zip(*result)
        self.state = np.concatenate(state)
        self.reward = np.array(reward)
        self.done = np.array(done)
        return self.state,self.reward,self.done


if __name__ == '__main__':
    # a=np.array([1,2,3,2,2,2,4])
    # b=np.where(a == 2)[0]
    # print(b,type(b),b.size)
    ENV_A=0.2
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 2), [0, 4], [0, 5], [4, 5], [3, 6], [2, 6], [5, 7], [5, 8]])
    g2 = nx.barabasi_albert_graph(20, 2)
    nx.draw_networkx(graph, with_labels=True)
    plt.show()
    # nx.draw(g2, with_labels=True)
    # plt.show()

    graph_list = [graph, graph]
    print(g2.nodes)
    print(graph_list)
    env1 = MP_Decycler(graph_list=graph_list, cpu_use=1,env_a=ENV_A)
    a, b, c = env1.get_state()
    print(a,b,c)
    print(env1.step(np.array([2, 4])))
    print(env1.step(np.array([4, 2])))
    # print(env1.step(np.array([-1, 2])))
    # print(env1.env_list[0].get_graph_state())
    a, b,c = env1.get_state()
    b=np.array(b)
    print(a, b, c)
