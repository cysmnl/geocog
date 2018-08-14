import os
import torch
import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar)

class ADNI(InMemoryDataset):

    def __init__(self, root, train=True, transform=None):
        super(ADNI, self).__init__(root, transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['train1.csv', 'test1.csv']

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        pass
        #print('no download avail..')

    def process(self):
        #connectome
        adj = np.loadtxt('../../cogd/model/graph/mci_connectome.csv',delimiter=',')
        a = coo_matrix(adj)

        #load
        train_path = "../../cogd/model/data/imaging/kfold/train1.csv"
        test_path  = "../../cogd/model/data/imaging/kfold/test1.csv"
        train = np.loadtxt(train_path,delimiter=',',skiprows=1)
        test = np.loadtxt(test_path,delimiter=',',skiprows=1)
        data_list = []
        
        for m in train:
            edge_index = torch.tensor([a.row,a.col], dtype=torch.long) # [2, num_edges] coo format
            edge_attr  = torch.tensor(np.transpose([a.data]), dtype=torch.float) # [num_edges, num_edge_features]
            node_feat = np.reshape(m[2:], (86,2), order='F')
            x = torch.tensor(node_feat, dtype=torch.float) # [num_nodes, num_node_features]
            y = torch.tensor([m[0]-1], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])