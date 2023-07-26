from models36_msd import GeometricEncoder
from all_util_monomer_batch36_msd import *
import os
import torch
import numpy as np
from data_process_batch_multi import GDataset
from torch_geometric.data import DataLoader

if os.path.isdir('pre_pdb_totrain'):
    pass
else:
    os.mkdir('pre_pdb_totrain')

TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
LR = 0.001  ###########0.001

NUM_EPOCHS = 1000
print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

model = GeometricEncoder(256)
dataset = 'monomer'
print('\nrunning on monomer stable')

optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
best_epoch = -1
best_pcc = -np.inf
model_file_name = 'modeldir_regression/regression_Gturb_msd4_nor.model'
result_file_name = 'result/result_regression_Gturb_msd4_nor.csv'

for epoch in range(NUM_EPOCHS):
    for single in range(2000):
        train_data = GDataset(root='data3', dataset='train_%s' % str(single))
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        train_regression(model=model, train_loader=train_loader,
                         optimizer=optimizer, epoch=epoch)
        if single % 50 == 0 and single != 0:
            print('predicting for valid data')
            G_list = torch.Tensor().cpu()
            P_list = torch.Tensor().cpu()
            for single in range(2000):
                valid_data = GDataset(root='data3', dataset='val_%s' % str(single))
                valid_loader = DataLoader(valid_data, batch_size=VALID_BATCH_SIZE, shuffle=False)
                G, P = predicting_regression(model, valid_loader)
                P_list = torch.cat((P_list, P), 0).clone().detach()
                G_list = torch.cat((G_list, G), 0).clone().detach()
            P_list = P_list.view(-1)
            G_list = G_list.view(-1)
            ret = pearson(P_list, G_list)
            if ret > best_pcc:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name, 'w') as f:
                    f.write('epoch,pcc\n')
                    f.write(str(epoch))
                    f.write(',')
                    f.write(str(ret))
                    f.write('\n')
                best_epoch = epoch
                best_pcc = ret
                print('mse improved at epoch ', best_epoch, 'pcc', best_pcc)
            else:
                print('No improvement since epoch ', best_epoch, 'pcc', ret, 'best', best_pcc)
