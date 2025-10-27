import numpy as np
import torch
from torch import optim
from copy import deepcopy

from .bc import BC

import sys
sys.path.append('..')
from src.models import Model
# from src.lunar_lander_utils import dynamics
import tqdm
# def dynamics_wrapper(a,b,env):
#     return dynamics(a,b,env)
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from models import Model
def CCIL(D_E, pi_0, dynamics, lr=3e-5, nsamp=4, pi_BC=None, sigma=4.05, wd=1e-4 , step = 5, hop = 1, device = 'cpu'):
    pi_init = deepcopy(pi_0).to(device)
    if pi_BC is None:
        pi_BC = BC(D_E, pi_0, lr=lr).to(device)
    print("Done w/ BC")
    # X_trajs = D_E[0]
    # U_trajs = D_E[1]
    X_trajs = [x[0] for x in D_E]
    U_trajs = [x[1] for x in D_E]
    X_IV = []
    U_BC = []
    for _ in range(nsamp):
        # 生成了基于专家策略的动作加噪声的动作轨迹
        # for i in range(len(X_trajs)):
            # index_0 = np.random.randint(0,len(X_trajs[i])-step-1)
            # noise_array = np.zeros(shape = (len(X_trajs[i])-1,1))
            # random_values = np.random.normal(size=(step,1))
            # noise_array[index_0:index_0+step] = random_values
            # U_BC.append(pi_BC(torch.from_numpy(X_trajs[i][:-1]).float()).detach().numpy()+sigma*noise_array)
        U_BC = [pi_BC(torch.from_numpy(xt[:-1]).float().to(device)).detach().cpu().numpy() + sigma * np.random.normal(size=(len(xt[:-1]), 1)) for xt in X_trajs]
        X_prime = np.concatenate([dynamics(X_trajs[i][:-hop], U_BC[i]) for i in range(len(D_E))], axis=0)
        X_IV.append(X_prime)  # samples from P(X|z)
    pi = pi_init
    # U_IV = np.array([ut for ut in U_trajs[1:]])
    U_IV = np.concatenate([ut[hop:] for ut in U_trajs], axis=0) # single-sample estimate of E[Y|z]
    if isinstance(pi, Model):
        optimizer = optim.Adam(pi.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.Adam(pi.parameters(), lr=lr)
    print('IV Data', X_IV[0].shape, U_IV.shape)
    # for step in tqdm.tqdm(range(int(5))):
    for step in tqdm.tqdm(range(int(5e4))):
    # for step in range(int(5e4)):
        idx = np.random.choice(len(X_IV[0]), 128)
        actions = torch.from_numpy(U_IV[idx]).to(device)
        optimizer.zero_grad()
        outputs_1 = 0
        outputs_2 = 0
        sample_idx = list(range(nsamp))
        np.random.shuffle(sample_idx)
        for i in range(int(nsamp / 2)):
            states_1 = torch.from_numpy(X_IV[sample_idx[i]][idx]).to(device)
            states_2 = torch.from_numpy(X_IV[sample_idx[i + int(nsamp / 2)]][idx]).to(device)
            with torch.no_grad():
                outputs_1 += pi(states_1.float())
            outputs_2 += pi(states_2.float())
        outputs_1 = (outputs_1 / (nsamp / 2)).detach()
        outputs_2 = outputs_2 / (nsamp / 2)
        factor_1 = (outputs_1 - actions.float()).detach()
        factor_2 = outputs_2 - actions.float()
        loss = torch.mean(factor_1 * factor_2).to(device)
        loss.backward()
        optimizer.step()
    return pi