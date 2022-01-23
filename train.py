from gnn import AllEmbedding, CatEmbedding, UV_Aggregator, UV_Encoder, Social_Aggregator, Social_Encoder, GraphRec, Combiner

import torch
import torch.nn as nn
import pickle
import math
from math import sqrt
import time
import datetime
import os
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import StandardScaler
import torch.utils.data
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae, best_medae, best_auc, args):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, label01_list, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), label01_list.to(device), labels_list.to(device), args.l2_lambda)
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %d] loss: %.3f, The best rmse/mae/best_medae: %.6f / %.6f / %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae, best_medae, best_auc))
            running_loss = 0.0
    return 0

def test(model, device, test_loader, ss):
    model.eval()
    tmp_pred = []
    target = []
    target01 = []
    u_ls = []
    v_ls = []
    with torch.no_grad():
        for test_u, test_v, tmp_target01, tmp_target in test_loader:
            u_ls.append(list(test_u.numpy()))
            v_ls.append(list(test_v.numpy()))
            target01.append(list(tmp_target01.numpy()))
            target.append(list(tmp_target.numpy()))
            test_u, test_v,tmp_target01, tmp_target = test_u.to(device), test_v.to(device),tmp_target01.to(device), tmp_target.to(device)
            _, val_output = model.forward(test_u, test_v)
            if ss:
                val_output = val_output.data.cpu().numpy()
                val_output = np.where(val_output<0,0,val_output)
                val_output = np.exp(val_output)
                tmp_pred.append(list(ss.inverse_transform(val_output)))
            else:
                tmp_pred.append(list(val_output.data.cpu().numpy()))
            
    tmp_pred = np.array(sum(tmp_pred, []))
    target01 = np.array(sum(target01, []))
    target = np.array(sum(target, []))
    u_ls = np.array(sum(u_ls, []))
    v_ls = np.array(sum(v_ls, []))
    expected_rmse = sqrt(mean_squared_error(target,tmp_pred))
    mae = mean_absolute_error(target,tmp_pred)
    medae = median_absolute_error(target,tmp_pred)
    auc = roc_auc_score(target01, tmp_pred)
    return expected_rmse, mae, medae, auc, u_ls, v_ls, target, tmp_pred



