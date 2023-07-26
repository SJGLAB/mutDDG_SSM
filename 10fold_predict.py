"""
-*- coding: utf-8 -*-

@Author : Shanshan Li
@Time : 2023/2/10 11:12
@File : 10折预测.py
"""
import os.path
from scipy.stats import pearsonr
import numpy as np
import xgboost as xgb
import pandas as pd
import joblib

sdir = '3213_nor_6144'
name = '3213raw'
test_name = 'S203'

dir = './' + sdir + '/' + name + '/'
if not os.path.exists('result/' + sdir):
    os.mkdir('result/' + sdir)

if test_name == 'sym':
    sym_X1 = np.load(sdir + '/sym_x_d.npy')
    sym_X2 = np.load(sdir + '/sym_x_r.npy')
    sym_Y1 = np.load(sdir + '/sym_y_d.npy')
    sym_Y2 = np.load(sdir + '/sym_y_r.npy')

elif test_name == 'S347':
    sym_X1 = np.load(sdir + '/S347_x_d.npy')
    sym_Y1 = np.load(sdir + '/S347_y_d.npy')

elif test_name == 'S203':
    sym_X1 = np.load(sdir + '/S203_x_d.npy')
    sym_Y1 = np.load(sdir + '/S203_y_d.npy')

scalar = joblib.load(sdir + '/3213_standard_scalar.joblib')
sym_X1 = scalar.transform(sym_X1)
dvalid1 = xgb.DMatrix(data=sym_X1)
pcc_list_d = []
pred_list_d = []
for i in range(10):
    model = xgb.Booster(model_file=dir + f'XGB_fold{i}.xgb')
    pred_d = model.predict(dvalid1)
    pred_list_d.append(list(pred_d))
    pcc_d_value = pearsonr(pred_d, sym_Y1)[0]
    pcc_list_d.append(pcc_d_value)

pcc_list_r = []
pred_list_r = []
if test_name == 'sym':
    sym_X2 = scalar.transform(sym_X2)
    dvalid2 = xgb.DMatrix(data=sym_X2)
    for i in range(10):
        model = xgb.Booster(model_file=dir + f'XGB_fold{i}.xgb')
        pred_r = model.predict(dvalid2)
        pred_list_r.append(list(pred_r))
        pcc_r_value = pearsonr(pred_r, sym_Y2)[0]
        pcc_list_r.append(pcc_r_value)

pred_ave_d = np.average(np.array(pred_list_d), axis=0)
pcc_d = pearsonr(pred_ave_d, sym_Y1)[0]
print('PCC direct:', pcc_d)
if test_name == 'sym':
    pred_ave_r = np.average(np.array(pred_list_r), axis=0)
    pcc_r = pearsonr(pred_ave_r, sym_Y2)[0]
    print('PCC reverse:', pcc_r)
if test_name != 'sym':
    with open('result/' + sdir + '/' + name + test_name + '_result_detail.csv', 'w') as f:
        f.write('y,pred\n')
        for i in range(len(pred_ave_d)):
            f.write(str(sym_Y1[i]))
            f.write(',')
            f.write(str(pred_ave_d[i]))
            f.write('\n')
else:
    with open('result/' + sdir + '/' + name + test_name + '_result_detail.csv', 'w') as f:
        f.write('y_d,pred_d,y_r,pred_r\n')
        for i in range(len(pred_ave_d)):
            f.write(str(sym_Y1[i]))
            f.write(',')
            f.write(str(pred_ave_d[i]))
            f.write(',')
            f.write(str(sym_Y2[i]))
            f.write(',')
            f.write(str(pred_ave_r[i]))
            f.write('\n')

