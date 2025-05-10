import os
from pathlib import Path
import pickle
from random import randint
import random
import networkx as nx
from torch.utils.data import Dataset, DataLoader
import shutil
from tqdm import tqdm


class GraphDataset(Dataset):
    def __init__(self, data_dir=None, size=None):
        assert data_dir is not None
        self.data_dir = data_dir
        self.graph_paths = sorted(list(self.data_dir.rglob("*.pickle")))
        if size is not None:
            assert size > 0
            self.graph_paths = self.graph_paths[:size]
        self.num_graphs = len(self.graph_paths)

    def __getitem__(self, idx):
        return read_nx_from_graph(self.graph_paths[idx])

    def __len__(self):
        return self.num_graphs


def read_nx_from_graph(graph_path):
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    return G


def gen_data(graph_type, num_graph, min_n, max_n, path, ba_m=3, seed=0):
    random.seed(seed)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"path '{path}' created.")
    else:
        print(f"path '{path}' already exists.")
    if graph_type == 'BA':
        for num_g in tqdm(range(num_graph)):
            n = randint(min_n, max_n)
            stub = f"GR_{graph_type}_{min_n}_{max_n}_{num_g}"
            g = nx.barabasi_albert_graph(n, ba_m)
            output_file = path / f"{stub}.pickle"
            with open(output_file, 'wb') as f:
                pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)
            print(f"Generated graph {stub}")
    else:
        raise NotImplementedError

def get_dataloader(train_batch_size=4,test_batch_size=-1):
    train_cache_directory = Path('/home/neu/sunxiaojie/gfnco/data/train')
    trainset = GraphDataset(train_cache_directory)

    test_cache_directory=Path('/home/neu/sunxiaojie/gfnco/data/test')
    testset = GraphDataset(test_cache_directory, size=500)
    if test_batch_size==-1:
        test_batch_size =len(testset)
    collate_fn = lambda graphs: list(graphs)
    train_loader = DataLoader(trainset, batch_size=train_batch_size,
                              shuffle=True, collate_fn=collate_fn, drop_last=False,
                              num_workers=5, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=test_batch_size,
                             shuffle=False, collate_fn=collate_fn, num_workers=1, pin_memory=True)
    return train_loader,test_loader
if __name__ == '__main__':

    path = Path('data/train01')
    # shutil.rmtree(path)  # 递归的删除对应文件夹和其中所有文件
    gen_data(graph_type='BA', num_graph=8, min_n=500, max_n=1000, path=path,ba_m=2,seed=0)
    # path = Path('data/test')
    # gen_data(graph_type='BA', num_graph=256, min_n=800, max_n=1000, path=path, ba_m=3, seed=0)
    # train_cache_directory = Path('data/train')
    # trainset = GraphDataset(train_cache_directory)
    # train_batch_size = 8
    # collate_fn = lambda graphs: list(graphs)
    # train_loader = DataLoader(trainset, batch_size=train_batch_size,
    #                           shuffle=True, collate_fn=collate_fn, drop_last=False,
    #                           num_workers=5, pin_memory=True)
    # print(train_loader)
    # print(len(trainset))
    # print(trainset.__getitem__(0))
    # for batch_idx, gbatch in enumerate(train_loader):
    #     print(batch_idx)
    #     print(gbatch)
    #     print(len(gbatch))
