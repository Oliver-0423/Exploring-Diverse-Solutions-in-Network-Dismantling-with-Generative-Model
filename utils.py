import os
import random
from multiprocessing import Pool

from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
from torch_geometric.data import Batch


class TransitionBuffer():
    def __init__(self,  size):
        self.size = size
        self.buffer = []
        self.pos = 0

    def reset(self):
        self.buffer = []
        self.pos = 0

    @staticmethod
    def batch_to_data(batch):
        return batch.to_data_list()

    def add_batch(self, batch):

        traj_s, traj_a, traj_r, traj_d = batch  # s,a,r,d
        traj_d = np.array(traj_d)
        traj_len = 1 + np.sum(~traj_d, axis=0)
        with Pool(processes=None) as p:
            data_list = list(p.map(self.batch_to_data, traj_s))
        data_list1 = list(zip(*data_list))
        k=len(data_list1)
        for i in range(k):  # batch 中第i个图
            n=traj_len[i] - 1
            for j in range(n):  # 第i个图第j个状态
                t1 = (
                    data_list1[i][j], traj_r[j][i], traj_a[j][i], data_list1[i][j + 1], traj_r[j + 1][i],
                    traj_d[j + 1][i])
                self.add_single_transition(t1)

    def add_single_transition(self, inp):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.pos] = inp
        self.pos = (self.pos + 1) % self.size

    def sample(self, batch_size):
        # random.sample: without replacement
        # i=1
        # batch_list=[]
        # while len(self.buffer) < batch_size:
        #     batch_size=batch_size/2
        #     i=i*2
        # for i in range(i):
        #     batch = random.sample(self.buffer,int(batch_size) ) # list of transition tuple
        #     batch_list=batch_list+batch
        batch = random.sample(self.buffer, batch_size)
        return self.transition_collate_fn(batch)

    def sample_from_indices(self, indices):
        batch = [self.buffer[i] for i in indices]
        return self.transition_collate_fn(batch)

    @staticmethod
    def transition_collate_fn(transition_ls):
        s_batch, logr_batch, a_batch, s_next_batch, logr_next_batch, d_batch = zip(*transition_ls)
        s_batch = Batch.from_data_list(s_batch)
        logr_batch = torch.tensor(logr_batch)
        a_batch = torch.tensor(a_batch)
        s_next_batch = Batch.from_data_list(s_next_batch)
        logr_next_batch = torch.tensor(logr_next_batch)
        d_batch = torch.tensor(d_batch)
        return  s_batch, logr_batch, a_batch, s_next_batch, logr_next_batch, d_batch

    def __len__(self):
        return len(self.buffer)


def pad_batch(vec, dim_per_instance, padding_value, dim=0, batch_first=True):
    tupllle = torch.split(vec, dim_per_instance, dim=dim)
    pad_tensor = pad_sequence(tupllle, batch_first=batch_first, padding_value=padding_value)
    return pad_tensor


def seed_all(seed, verbose=True):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    if verbose:
        print("==> Set seed to {:}".format(seed))
