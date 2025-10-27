import numpy as np
import torch
from torch import optim, nn

import sys
sys.path.append('..')
from src.models import Model

import tqdm

def BC(D_E, pi_0, loss_fn=torch.nn.MSELoss(), lr=3e-4, steps=int(5e4), wd=1e-3, device = 'cpu'):
    pi = pi_0.to(device) 
    X = np.concatenate([x[0] for x in D_E], axis=0)
    U = np.concatenate([x[1] for x in D_E], axis=0) 
    print('BC Data', X.shape, U.shape)
    if isinstance(pi, Model):
        optimizer = optim.Adam(pi.parameters(), lr=lr, weight_decay=wd) 
    else: 
        optimizer = optim.Adam(pi.parameters(), lr=lr)
    for step in tqdm.tqdm(range(steps)):
        idx = np.random.choice(len(X), 128)
        states = torch.from_numpy(X[idx]).to(device)
        actions = torch.from_numpy(U[idx]).to(device)
        optimizer.zero_grad()
        outputs = pi(states.float()) # states itemsize=8
        loss = loss_fn(outputs, actions.float().to(device))
        loss.backward() 
        optimizer.step() 
    return pi 
