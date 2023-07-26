from math import sqrt
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, r2_score, matthews_corrcoef
import os
import torch.nn as nn


def rmse(g, p):
    _g = (g - min(g)) / (max(g) - min(g))
    _p = (p - min(g)) / (max(g) - min(g))
    rmse = sqrt(((_g - _p) ** 2).mean(axis=0))
    return rmse


def mse(g, p):
    _g = (g - min(g)) / (max(g) - min(g))
    _p = (p - min(g)) / (max(g) - min(g))
    mse = ((_g - _p) ** 2).mean(axis=0)
    return mse


def msd(coords1, coords2):
    total = 0
    for i in range(len(coords1)):
        total += (float(coords1[i]) - float(coords2[i])) ** 2
    msd = total / len(coords1)
    return torch.tensor([msd])


def calculate_metric(gt, pred):
    pred[pred > 0.5] = 1
    pred[pred < 1] = 0
    confusion = confusion_matrix(gt, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print('Accuracy:', (TP + TN) / float(TP + TN + FP + FN))
    print('Sensitivity:', TP / float(TP + FN))
    print('Specificity:', TN / float(TN + FP))
    return (TP + TN) / float(TP + TN + FP + FN), TP / float(TP + FN), TN / float(TN + FP)


def get_roc_auc(y, p):
    return roc_auc_score(y, p)


from scipy.stats import pearsonr


def pearson(y, p):
    return pearsonr(y, p)[0]


def r2(y, p):
    return r2_score(y, p)


def process_fea(pdbname, mutinfo):
    try:
        pdbdf = 'pre_rsa/' + str(pdbname) + '_pre.csv'
        savefilecont = gen_graph_data(pdbdf, mutinfo, cutoff=3)
    except:
        savefilecont = None
    return savefilecont





def gen_graph_data(pdbfile, mutinfo, cutoff):
    max_dis = 12
    sample = build_pt_graph(pdbfile, mutinfo, cutoff, max_dis)
    return sample

def get_symbol(symbol):
    atomnames = ['C', 'N', 'O', 'S']
    if symbol in atomnames:
        c = 1
    else:
        c =None
    return c

def build_pt_graph(pdbfile, mutinfo, cutoff=3, max_dis=12):
    atomnames = ['C', 'N', 'O', 'S']
    residues = ['ARG', 'MET', 'VAL', 'ASN', 'PRO', 'THR', 'PHE', 'ASP', 'ILE', \
                'ALA', 'GLY', 'GLU', 'LEU', 'SER', 'LYS', 'TYR', 'CYS', 'HIS', 'GLN', 'TRP']
    res_code = ['R', 'M', 'V', 'N', 'P', 'T', 'F', 'D', 'I', \
                'A', 'G', 'E', 'L', 'S', 'K', 'Y', 'C', 'H', 'Q', 'W']
    struct_code = ['H', 'B', 'E', 'G', 'S', 'T', '-', 'I']  #######
    res2code = {x: idxx for x, idxx in zip(residues, res_code)}
    atomdict = {x: i for i, x in enumerate(atomnames)}
    resdict = {x: i for i, x in enumerate(residues)}
    struct2code = {x: i for i, x in enumerate(struct_code)}
    V_atom = len(atomnames)
    V_res = len(residues)
    V_structs = len(struct_code)
    df = pd.read_csv(pdbfile)
    df = df.reset_index(drop=True)
    if mutinfo is not None:
        chain_mut = df[df['chain_id'] == mutinfo.split('_')[0]]
        residue_mut = chain_mut[chain_mut['residue_number'] == int(mutinfo.split('_')[1])]
        residue_mut = residue_mut.reset_index(drop=True)
        residue_mut['cnos'] = residue_mut['element_symbol'].apply(get_symbol)
        residue_mut = residue_mut.dropna(subset=['cnos'])
        cd_tensor = residue_mut.loc[:, ['x_coord', 'y_coord', 'z_coord']]
        mut_coors_matrix = torch.from_numpy(cd_tensor.values)

    n_features = V_atom + V_res + V_structs + 4
    atoms = []
    coords_all = []
    df['cnos'] = df['element_symbol'].apply(get_symbol)
    df = df.dropna(subset=['cnos'])
    df = df.reset_index(drop=True)
    for i in range(len(df)):
        features = [0] * n_features
        atomname = df.loc[i, 'atom_name']
        elemname = list(filter(lambda x: x.isalpha(), atomname))[0]
        resname = df.loc[i, 'residue_name']
        chainid = df.loc[i, 'chain_id']
        res_idx = df.loc[i, 'residue_number']
        x = df.loc[i, 'x_coord']
        y = df.loc[i, 'y_coord']
        z = df.loc[i, 'z_coord']
        sasa = df.loc[i, 'charge']
        struct = df.loc[i, 'segment_id']
        if elemname not in atomdict:
            continue
        coords = torch.tensor([x, y, z])
        atomid = atomdict[elemname]

        resid = resdict[resname]
        features[atomid] = 1
        features[V_atom + resid] = 1

        cr_token = '{}_{}'.format(chainid, res_idx)
        float_cd = [float(x) for x in coords]
        cd_tensor = torch.tensor(float_cd)
        # 24
        if cr_token == mutinfo:
            features[V_atom + V_res] = 1

        # 25-32
        struct_id = struct2code[struct]
        features[V_atom + V_res + struct_id + 1] = 1

        # 27>33
        if cr_token != mutinfo:
            features[V_atom + V_res + V_structs + 1] = 1

        # 28>34
        if atomname == 'CA':
            features[V_atom + V_res + V_structs + 2] = 1
        flag = False
        dissss = torch.norm(cd_tensor - mut_coors_matrix, dim=1)
        flag = (dissss < max_dis).any()
        # 35
        if float(sasa) == 0.:
            features[V_atom + V_res + V_structs + 3] = 0
        elif float(sasa) == -1.:
            features[V_atom + V_res + V_structs + 3] = -1
        else:
            features[V_atom + V_res + V_structs + 3] = 1

        if flag:
            atoms.append(features)
            coords_all.append(float_cd)

    atoms = torch.tensor(atoms, dtype=torch.float)
    N = atoms.size(0)
    atoms_type = torch.argmax(atoms[:, :4], 1)
    atoms_type = atoms_type.unsqueeze(1).repeat(1, N)
    edge_type = atoms_type * 4 + atoms_type.t()

    pos = torch.Tensor(coords_all)
    row = pos[:, None, :].repeat(1, N, 1)
    col = pos[None, :, :].repeat(N, 1, 1)
    direction = row - col
    del row, col
    distance = torch.sqrt(torch.sum(direction ** 2, 2)) + 1e-10
    distance1 = (1.0 / distance) * (distance < float(cutoff))*(distance>1e-10).float()####
    del distance
    diag = torch.diag(torch.ones(N))
    dist = diag + (1 - diag) * distance1
    del distance1, diag
    flag = (dist > 0).float()
    direction = direction * flag.unsqueeze(2)
    del direction, dist
    edge_sparse = torch.nonzero(flag)  # K,2
    edge_attr_sp = edge_type[edge_sparse[:, 0], edge_sparse[:, 1]]

    savefilecont = [atoms, edge_sparse, edge_attr_sp, mut_coors_matrix]

    return savefilecont


import torch

def sum_msd(a, b):
    msd = torch.norm(a - b, p=2,dim=1)
    return msd


def msd_loss(pred, y):
    msd = torch.norm(pred - y, p=2) / len(y) * 10000.
    print(y.shape)
    return msd


def msd_val(pred, y):
    msd = torch.norm(pred - y, p=2) / len(y) * 10000.
    return msd


def train_regression(model, train_loader, optimizer, epoch):
    model.train()
    model = model.cuda()
    for idx, data in enumerate(train_loader):
        data = data.to('cuda')
        a1 = model.regre(data).view(-1,1)
        y = sum_msd(data.y,data.coords).view(-1,1)
        optimizer.zero_grad()
        loss = nn.MSELoss()(a1, y)  ##############
        print('loss:', loss.item())
        loss.backward()
        loss.requires_grad_()
        optimizer.step()
        print('Train step: {} \tLoss: {:.6f}'.format((epoch) * (len(train_loader.dataset))/128 + idx, loss.item()))


def predicting_regression(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    model = model.cuda()
    # res1 = 0
    total_preds = total_preds.to('cuda')
    total_labels = total_labels.to('cuda')
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to('cuda')
            a1 = model.regre(data).view(-1,1)
            y = sum_msd(data.y, data.coords).view(-1,1)
            total_preds = torch.cat((total_preds, a1), 0).clone().detach()
            total_labels = torch.cat((total_labels, y), 0).clone().detach()
    return total_labels.cpu(), total_preds.cpu()


def get_mcc(g, p):
    return matthews_corrcoef(g, p)
