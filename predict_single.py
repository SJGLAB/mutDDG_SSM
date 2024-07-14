"""
-*- coding: utf-8 -*-

@Author : Shanshan Li
@Time : 2023/6/20 11:15
@File : predict_single.py
"""
########################################
name = '1AMQ'
chain = 'A'
position = '191'
wildtype = 'C'
muttype = 'S'
rosetta_file = '/mnt/d/software/ubuntu_software/rosetta_bin_linux_2021.16.61629_bundle/main/source/bin/relax.static.linuxgccrelease'
start_struct = 'example/1amqA_relaxed.pdb'
struct = '1amqA_relaxed.pdb'
variant_resfile = 'example/' + name + '_' + wildtype + position + muttype + '/nataa_mutations.resfile'
dir = '3213_nor_6144/3213raw/'
########################################
import os
import warnings
warnings.filterwarnings('ignore')
if not os.path.exists('example/' + name + '_' + wildtype + position + muttype):
    os.mkdir('example/' + name + '_' + wildtype + position + muttype)
with open('example/' + name + '_' + wildtype + position + muttype + '/nataa_mutations.resfile', 'w') as f:
    f.write('NATAA\n')
    f.write('start\n')
    f.write(str(position))
    f.write(' ')
    f.write(str(chain))
    f.write(' PIKAA ')
    f.write(str(muttype))
    f.write('\n')

rosetta_relax_cmd = ' '.join([rosetta_file,
                              '-in:file:s', start_struct, '-in:file:fullatom',
                              '-relax:constrain_relax_to_start_coords',
                              '-out:no_nstruct_label', '-relax:ramp_constraints false',
                              '-relax:respect_resfile',
                              '-packing:resfile', variant_resfile,
                              '-default_max_cycles', '200',
                              '-out:file:scorefile', os.path.join('example', name + '_relaxed.sc'),
                              '-out:path:pdb', 'example/'+ name + '_' + wildtype + position + muttype
                              ])

os.system(rosetta_relax_cmd)

from biopandas.pdb import PandasPdb
from Bio.PDB import PDBParser, SASA
from Bio.PDB.DSSP import DSSP

def jisuansasa(struct, chain_id, resid, atomname):
    try:
        a = struct[0][chain_id][resid][atomname].sasa
    except:
        a = 0.
    return float(a)


def charge_raw(file):
    try:
        # file = pdbname
        df1 = PandasPdb().read_pdb(file).df['ATOM']
        pdb1 = PDBParser().get_structure(file=file, id=None)
        print(pdb1)
        sr1 = SASA.ShrakeRupley()
        sr1.compute(pdb1, level='S')
        model = pdb1[0]
        df1['charge'] = df1.apply(lambda x: jisuansasa(pdb1, x['chain_id'], x['residue_number'], x['atom_name']),
                                  axis=1)
        dssp1 = DSSP(model, file, dssp='mkdssp')
        df1['segment_id'] = df1.apply(lambda x: dssp1[(x['chain_id'], x['residue_number'])][2], axis=1)
        df1['b_factor'] = df1.apply(lambda x: dssp1[(x['chain_id'], x['residue_number'])][3], axis=1)
        print(2)
        df1.to_csv(file + '.csv', index=False)
        a = 1
        print('suc:', file)
    except:
        a = None
        print('error:', file)
    return a


charge_raw(start_struct)
charge_raw('example/' + name + '_' + wildtype + position + muttype + '/' + start_struct.split('/')[-1])

from models36_msd import GeometricEncoder
import torch
import xgboost as xgb
from all_util_monomer_batch36_msd import gen_graph_data
import joblib
import numpy as np
def gen_fea(wildtypefile, muttypefile, mutinfo):
    try:
        A, E, Ea, _m = gen_graph_data(wildtypefile, mutinfo, 3)
        A_m, E_m, Ea_m, _mm = gen_graph_data(muttypefile, mutinfo, 3)
        # A, E, Ea = A.to('cuda'), E.to('cuda'), Ea.to('cuda')
        # A_m, E_m, Ea_m = A_m.to('cuda'), E_m.to('cuda'), Ea_m.to('cuda')
        # print(True)
        model = GeometricEncoder(256)
        gnnfile = 'modeldir_regression/regression_Gturb_msd4_nor.model'  #######################
        model.load_state_dict(torch.load(gnnfile,map_location='cpu'))
        # model.cuda()
        with torch.no_grad():
            fea = model.gen_features(A, E, Ea, A_m, E_m, Ea_m)
            fea = fea.cpu().numpy().flatten()
            print('fea', fea)
    except:
        fea = None
        # print(1)
    return fea


fea = gen_fea(start_struct + '.csv', 'example/' + name + '_' + wildtype + position + muttype + '/' + struct+'.csv',chain+'_'+position)
scalar = joblib.load('3213_nor_6144/3213_standard_scalar.joblib')
fea = np.reshape(fea,newshape=(1,-1))
fea = scalar.transform(fea)
dvalid1 = xgb.DMatrix(data=fea)
res = 0
for i in range(10):
    model = xgb.Booster(model_file=dir + f'XGB_fold{i}.xgb')
    res += model.predict(dvalid1)

result = float(res / 10)
print('pdbname:', name)
print('position:', chain + '_' + position)
print('mutant:', muttype)
print('ddg:', result)
