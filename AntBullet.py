from stable_baselines3 import SAC # version-2.2.1
import pybullet_envs
import numpy as np
import torch
import pandas as pd

from src.pybullet_utils import CnfndWrapper, rollout, noisy_rollout, T, dynamics

from src.models import Model
from src.learners.bc import BC
from learners.ccil_antbullet import CCIL,dynamics_wrapper
from src.learners.ccil2 import CCIL2
from src.TCN_AIT_test import TCN_test

import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy
import gym,pickle,time

from src.pybullet_utils import CnfndWrapper
env = CnfndWrapper(gym.make("AntBulletEnv-v0"))
e = 'ant'

data_version = "3_TCN_CW"
method = 'C2L'

TCN = [True if "3_TCN" in data_version else False][0]
AIT = [True if method == "C2L" else False][0]

from stable_baselines3 import SAC
expert_net = SAC.load("./src/experts/ant_expert_local")
def expert(s):
    return expert_net.predict(s, deterministic=True)

env.reset()
   
denoised_expert_trajs = []
Js = []
for _ in range(10):
    s_traj, a_traj, J = rollout(expert, env)
    denoised_expert_trajs.append((s_traj, a_traj))
    Js.append(J)
# print(Js, np.mean(Js))

env.reset()
expert_trajs2 = []
Js = []
for _ in range(25):
    s_traj, a_traj, J = noisy_rollout(expert, env, sigma=2, full_state=True, n = 500, distribution= 'normal', hop = 3, TCN = TCN)
    expert_trajs2.append((s_traj, a_traj))
    Js.append(J)

print(np.mean(Js))

# Generate test set
env.reset()
denoised_test_s = []
denoised_test_a = []
P = []
V = []
C = []

for _ in range(100):
    s_traj, _, J, p, v, c = rollout(expert, env, full_state=True)
    print(J)
    denoised_test_s.append(s_traj)
    denoised_test_a.append(expert(s_traj)[0])
    P.append(p)
    V.append(v)
    C.append(c)

with open("data/{0}_train_data/denoised_test_{0}_{1}.pkl".format(e, data_version), "wb") as f:
     data = {"s": denoised_test_s,"a": denoised_test_a,"P": P,"V": V,"C": C}
     pickle.dump(data,f) 

noisy_test_s = []
noisy_test_a = []
P = []
V = []
C = []
for _ in range(100):
    s_traj, _, J, p, v, c = noisy_rollout(expert, env, sigma=2, full_state=True, n = 500, distribution= 'normal', hop = 3, TCN = TCN)
    print(J)
    noisy_test_s.append(s_traj)
    noisy_test_a.append(expert(s_traj)[0])
    P.append(p)
    V.append(v)
    C.append(c)

with open("data/{0}_train_data/noisy_test_{0}_{1}.pkl".format(e, data_version), "wb") as f:
     data = {"s": noisy_test_s,"a": noisy_test_a,"P": P,"V": V,"C": C}
     pickle.dump(data,f)

#  Generate training sets
for i in range(0, 5):
    for size in [10, 20, 30, 40, 50]:
        print(i, size)
        s_trajs = []
        a_trajs = []
        P = []
        V = []
        C = []
        for j in range(size):
            print(j)
            s_traj, a_traj, _, p, v, c, = noisy_rollout(expert, env, sigma=2, full_state=True,n = 500, distribution= 'normal', hop = 3, TCN = TCN)
            s_trajs.append(s_traj)
            a_trajs.append(a_traj)
            P.append(p)
            V.append(v)
            C.append(c)

        with open("data/{0}_train_data/train_{1}_{2}_{0}_{3}.pkl".format(e, size, i, data_version), "wb") as f:
            data = {"s": s_trajs,"a": a_trajs,"P": P,"V": V,"C": C}
            pickle.dump(data,f) 
      


with open("./data/{0}_train_data/denoised_test_{0}_{1}.pkl".format(e, data_version), "rb") as f:
    denoised_test_data = pickle.load(f)
denoised_test_s = denoised_test_data["s"]
denoised_test_a = denoised_test_data["a"]
with open("./data/{0}_train_data/noisy_test_{0}_{1}.pkl".format(e, data_version), "rb") as f:
    noisy_test_data = pickle.load(f)
noisy_test_s = noisy_test_data["s"]
noisy_test_a = noisy_test_data["a"]

denoised_test = [[denoised_test_s[i], denoised_test_a[i]] for i in range(len(denoised_test_a))]
noisy_test = [[noisy_test_s[i], noisy_test_a[i]] for i in range(len(noisy_test_a))]

def mse(pi, dataset):
    total = 0
    for (s_traj, a_traj) in dataset:
        total += np.linalg.norm(pi(s_traj) - a_traj)
    return total / len(dataset)

def eval_policy(pi, env, noisy=False):
    Js = []
    for _ in range(100):
        if noisy:
            s_traj, a_traj, J = noisy_rollout(pi, env, sigma = 2, n = 500, distribution= 'normal', hop = 3, TCN = TCN)
        else:
            s_traj, a_traj, J = rollout(pi, env)
        Js.append(J)
    return np.mean(Js)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bc_mse_noisy = []
    bc_mse_denoised = []
    bc_J_noisy = []
    bc_J_denoised = []

    CCIL_mse_noisy = []
    CCIL_mse_denoised = []
    CCIL_J_noisy = []
    CCIL_J_denoised = []

    CCIL2_mse_noisy = []
    CCIL2_mse_denoised = []
    CCIL2_J_noisy = []
    CCIL2_J_denoised = []

    for i in range(0, 5):
        bc_mse_noisy_i = []
        bc_mse_denoised_i = []
        bc_J_noisy_i = []
        bc_J_denoised_i = []

        CCIL_mse_noisy_i = []
        CCIL_mse_denoised_i = []
        CCIL_J_noisy_i = []
        CCIL_J_denoised_i = []

        CCIL2_mse_noisy_i = []
        CCIL2_mse_denoised_i = []
        CCIL2_J_noisy_i = []
        CCIL2_J_denoised_i = []

        for size in [10, 20, 30, 40, 50]:

            with open("./data/{0}_train_data/train_{1}_{2}_{0}_{3}.pkl".format(e, size, i, data_version),"rb") as f :
                all_traj = pickle.load(f)
            s_trajs = all_traj["s"]
            a_trajs = all_traj["a"]
            p_trajs = all_traj["P"]
            v_trajs = all_traj["V"]
            c_trajs = all_traj["C"]
            expert_trajs = [[s_trajs[i], a_trajs[i], p_trajs[i], v_trajs[i], c_trajs[i]] for i in range(len(s_trajs))]
            
            if AIT:
                hop = TCN_test(expert_trajs_data=expert_trajs)
                print("hop = ", hop)
            else:
                hop = 1
            
            pi_BC = BC(expert_trajs, Model(env.observation_space.shape[0], env.action_space.shape[0]), wd=5e-5, device = device)
            bc_mse_noisy_i.append(mse(lambda s: pi_BC(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), noisy_test))
            bc_mse_denoised_i.append(mse(lambda s: pi_BC(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), denoised_test))
            bc_J_noisy_i.append(eval_policy(lambda s: pi_BC(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), env, noisy=True))
            bc_J_denoised_i.append(eval_policy(lambda s: pi_BC(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), env, noisy=False))
            print('BC',
                size,
                bc_mse_noisy_i[-1],
                bc_mse_denoised_i[-1],
                bc_J_noisy_i[-1], 
              bc_J_denoised_i[-1])
            pi_CCIL = CCIL(expert_trajs,
                            Model(env.observation_space.shape[0], env.action_space.shape[0]),
                            dynamics_wrapper, pi_BC=pi_BC, wd=5e-3,env = env,device=device, workers = size,hop = hop)
            CCIL_mse_noisy_i.append(mse(lambda s: pi_CCIL(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), noisy_test))
            CCIL_mse_denoised_i.append(mse(lambda s: pi_CCIL(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), denoised_test))
            CCIL_J_noisy_i.append(eval_policy(lambda s: pi_CCIL(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), env, noisy=True))
            CCIL_J_denoised_i.append(eval_policy(lambda s: pi_CCIL(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), env, noisy=False))
            print('CCIL',
                size,
                CCIL_mse_noisy_i[-1],
                CCIL_mse_denoised_i[-1],
                CCIL_J_noisy_i[-1], 
                CCIL_J_denoised_i[-1])
            
            pi_CCIL2 = CCIL2(expert_trajs,
                        Model(env.observation_space.shape[0], env.action_space.shape[0]),
                        Model(env.observation_space.shape[0], env.action_space.shape[0]), lr = 5e-8 ,wd=5e-5, bc_reg=0, device = device, hop = hop)
            CCIL2_mse_noisy_i.append(mse(lambda s: pi_CCIL2(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), noisy_test))
            CCIL2_mse_denoised_i.append(mse(lambda s: pi_CCIL2(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), denoised_test))
            CCIL2_J_noisy_i.append(eval_policy(lambda s: pi_CCIL2(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), env, noisy=True))
            CCIL2_J_denoised_i.append(eval_policy(lambda s: pi_CCIL2(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), env, noisy=False))
            print('CCIL2',
                size,
                CCIL2_mse_noisy_i[-1],
                CCIL2_mse_denoised_i[-1],
                CCIL2_J_noisy_i[-1], 
                CCIL2_J_denoised_i[-1])
            
        bc_mse_noisy.append(bc_mse_noisy_i)
        bc_mse_denoised.append(bc_mse_denoised_i)
        bc_J_noisy.append(bc_J_noisy_i)
        bc_J_denoised.append(bc_J_denoised_i)


        with open("data/output_mse&J/{0}/bc_mse_noisy_{0}_{1}.pkl".format(e, method),"wb") as f:
            pickle.dump(bc_mse_noisy,f)
        with open("data/output_mse&J/{0}/bc_mse_denoised_{0}_{1}.pkl".format(e, method),"wb") as f:
            pickle.dump(bc_mse_denoised,f)
        with open("data/output_mse&J/{0}/bc_J_noisy_{0}_{1}.pkl".format(e, method),"wb") as f:
            pickle.dump(bc_J_noisy,f)
        with open("data/output_mse&J/{0}/bc_J_denoised_{0}_{1}.pkl".format(e, method),"wb") as f:
            pickle.dump(bc_J_denoised,f)   
        
        CCIL_mse_noisy.append(CCIL_mse_noisy_i)
        CCIL_mse_denoised.append(CCIL_mse_denoised_i)
        CCIL_J_noisy.append(CCIL_J_noisy_i)
        CCIL_J_denoised.append(CCIL_J_denoised_i)
        

        with open("data/output_mse&J/{0}/CCIL_mse_noisy_{0}_{1}.pkl".format(e, method),"wb") as f :
            pickle.dump(CCIL_mse_noisy,f)
        with open("data/output_mse&J/{0}/CCIL_mse_denoised_{0}_{1}.pkl".format(e, method),"wb") as f :
            pickle.dump(CCIL_mse_denoised,f)
        with open("data/output_mse&J/{0}/CCIL_J_noisy_{0}_{1}.pkl".format(e, method),"wb") as f :
            pickle.dump(CCIL_J_noisy,f)
        with open("data/output_mse&J/{0}/CCIL_J_denoised_{0}_{1}.pkl".format(e, method),"wb") as f :
            pickle.dump(CCIL_J_denoised,f)
            
        CCIL2_mse_noisy.append(CCIL2_mse_noisy_i)
        CCIL2_mse_denoised.append(CCIL2_mse_denoised_i)
        CCIL2_J_noisy.append(CCIL2_J_noisy_i)
        CCIL2_J_denoised.append(CCIL2_J_denoised_i)

        with open("data/output_mse&J/{0}/CCIL2_mse_noisy_{0}_{1}.pkl".format(e, method),"wb") as f :
            pickle.dump(CCIL2_mse_noisy,f)
        with open("data/output_mse&J/{0}/CCIL2_mse_denoised_{0}_{1}.pkl".format(e, method),"wb") as f :
            pickle.dump(CCIL2_mse_denoised,f)
        with open("data/output_mse&J/{0}/CCIL2_J_noisy_{0}_{1}.pkl".format(e, method),"wb") as f :
            pickle.dump(CCIL2_J_noisy,f)
        with open("data/output_mse&J/{0}/CCIL2_J_denoised_{0}_{1}.pkl".format(e, method),"wb") as f :
            pickle.dump(CCIL2_J_denoised,f)
