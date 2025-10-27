import numpy as np
import torch
from torch import optim
from copy import deepcopy
from functools import partial
from .bc import BC

import sys
sys.path.append('..')
from src.models import Model
from src.pybullet_utils import dynamics
import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def samp(j, dynamics, P, V, C, U_BC, U_trajs,env):
    return dynamics(P[j][:-1], V[j][:-1], C[j][:-1], U_BC[j][:-1], U_trajs[j][:-1],env = env)

def dynamics_wrapper(a, b, c, d, e, env):
    return dynamics(a, b, c, d, e, env)  

def CCIL(D_E, pi_0, dynamics, lr=3e-7, nsamp=4, pi_BC=None, wd=5e-3, sigma=6,env=None):
    pi_init = deepcopy(pi_0)
    if pi_BC is None:
        pi_BC = BC(D_E, pi_0, lr=lr)
    print("Done w/ BC")
    X_trajs = [x[0] for x in D_E] 
    U_trajs = [x[1] for x in D_E] 
    P = [x[2] for x in D_E]
    V = [x[3] for x in D_E]
    C = [x[4] for x in D_E]

    X_IV = []
    workers = 8
    with ProcessPoolExecutor(max_workers=workers) as executor:
        U_BC = [pi_BC(torch.from_numpy(xt[:-1]).float()).detach().cpu().numpy()  + sigma * np.random.normal(size=(len(xt[:-1]), 1)) for xt in X_trajs]
        samp_partial = partial(samp, dynamics=dynamics_wrapper, P=P, V=V, C=C, U_BC=U_BC, U_trajs=U_trajs,env = env)
        for i in range(nsamp):
            results = list(executor .map(samp_partial, range(len(D_E))))
            X_IV.append(np.concatenate(results, axis=0)) 
        
    pi = pi_init
    U_IV = np.concatenate([ut[1:] for ut in U_trajs], axis=0) 
    if isinstance(pi, Model):
        optimizer = optim.Adam(pi.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.Adam(pi.parameters(), lr=lr)
    print('IV Data', X_IV[0].shape, U_IV.shape)
    for step in tqdm.tqdm(range(int(5e4))):
        idx = np.random.choice(len(X_IV[0]), 128)
        actions = torch.from_numpy(U_IV[idx])
        optimizer.zero_grad()
        outputs_1 = 0
        outputs_2 = 0
        sample_idx = list(range(nsamp))
        np.random.shuffle(sample_idx)
        for i in range(int(nsamp / 2)):
            states_1 = torch.from_numpy(X_IV[sample_idx[i]][idx])
            states_2 = torch.from_numpy(X_IV[sample_idx[i + int(nsamp / 2)]][idx])
            with torch.no_grad():
                outputs_1 += pi(states_1.float())
            outputs_2 += pi(states_2.float())
        outputs_1 = (outputs_1 / (nsamp / 2)).detach()
        outputs_2 = outputs_2 / (nsamp / 2)
        factor_1 = (outputs_1 - actions.float()).detach()
        factor_2 = outputs_2 - actions.float()
        loss = torch.mean(factor_1 * factor_2)
        loss.backward()
        optimizer.step()
    return pi
