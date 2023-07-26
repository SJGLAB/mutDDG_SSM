import os
import pandas as pd
import os
from torch_geometric.data import InMemoryDataset, DataLoader, Dataset
from torch_geometric import data as DATA
import torch



class GDataset(InMemoryDataset):
    def __init__(self, root=None, dataset=None,
                 df=None, x=None, edge_sparse=None, edge_attr_sp=None, y=None,coords=None,
                 transform=None, pre_transform=None, pre_filter=None):

        super(GDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])

        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))

            self.process(df)
            self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def _download(self):
        pass

    def process(self, df):
        data_list = []
        data_len = len(df)
        for i in range(data_len):
            print('constructing %sth data'%str(i))
            feature = df.loc[i,'feature']
            atoms = torch.Tensor(feature[0].numpy()).view(-1,36)
            n_size = len(atoms)
            edge_sparse = torch.Tensor(feature[1].numpy()).view(-1,2).long()
            edge_attr_sp = torch.Tensor(feature[2].numpy()).view(-1,1).float()
            coords = torch.Tensor(feature[3].numpy()).view(-1,3)
            y = torch.tensor(df.loc[i,'y']).view(-1,3).float()
            GData = DATA.Data(x=atoms, edge_sparse=edge_sparse, edge_attr_sp=edge_attr_sp,
                              coords=coords,y=y)
            GData.__setitem__('c_size',torch.LongTensor(n_size))
            data_list.append(GData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('data construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

