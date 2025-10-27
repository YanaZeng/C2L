import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy
import pickle,time

# LunarLander
import gym

from src.lunar_lander_env import LunarLanderContinuous
from src.lunar_lander_utils import T, dynamics, rollout, noisy_rollout, eval_policy

from stable_baselines3 import PPO,SAC
import argparse

from src.lunar_lander_env import LunarLanderContinuous
from src.models import Model, LinearModel
from src.oadam import OAdam, add_weight_decay, net_to_list
from src.learners.bc import BC
from src.learners.ccil import CCIL
from src.learners.ccil2 import CCIL2
from src.TCN_AIT_test import TCN_test
# LQG System
import src.lqg

A = np.array([[1., 0.1], [0, 1.]])

data_version = "1_TCN"
# method = 'C2L'
method = 'Raw'
e = 'll'

TCN = [True if "3_TCN" in data_version else False][0]
AIT = [True if method == "C2L" else False][0]


env = LunarLanderContinuous(confounding=False, fixed_terrain=True)

expert_net = PPO.load("src\experts\ll_expert_curr")

K_star = src.lqg.solve(src.lqg.A, src.lqg.B, src.lqg.Q, src.lqg.R)
print(K_star)
def expert(s):
    if s.size == 2:
        return K_star @ s
    else:
        return s @ K_star.T
    
def learner(K, s):
    if s.size == 2:
        return K @ s
    else:
        return s @ K.T

def expert(s):
    return expert_net.predict(s, deterministic=True)


env.reset()


# # Generate test set
# env.reset()
# denoised_test = []
# Js = []
# for _ in range(100):
#     s_traj, _, J = rollout(expert, env)
#     Js.append(J)
#     denoised_test.append((s_traj, expert(s_traj)[0]))
# print(np.mean(Js)) 

# with open("data/{0}_train_data/denoised_test_{0}_{1}.pkl".format(e, data_version), "wb") as f:
#     pickle.dump(denoised_test, f)


# noisy_test = []
# Js = []
# for _ in range(100): 
#     s_traj, _, J = noisy_rollout(expert, env, sigma = 5,step = 100, hop = 3, distribution = 'normal', TCN = TCN)
#     # print(J)
#     Js.append(J)
#     noisy_test.append((s_traj, expert(s_traj)[0]))
# print(np.mean(Js))
# with open("data/{0}_train_data/noisy_test_{0}_{1}.pkl".format(e, data_version), "wb") as f:
#     pickle.dump(noisy_test, f)

# # Generate training sets
# for i in range(0, 5):
#     for size in [10, 20, 30, 40, 50]:
#         print('Training dataset：i = ', i,'size = ', size)
#         trajs = []
#         for _ in range(size):
#             s_traj, a_traj, _ = noisy_rollout(expert, env, sigma=5,step=100, hop = 3, distribution = 'normal', TCN = TCN)
#             trajs.append((s_traj, a_traj))
#         with open("data/{0}_train_data/train_{1}_{2}_{0}_{3}.pkl".format(e, size, i, data_version), "wb") as f:
#             pickle.dump(trajs, f)


with open("./data/{0}_train_data/denoised_test_{0}_{1}.pkl".format(e, data_version), "rb") as f:
    denoised_test = pickle.load(f)
with open("./data/{0}_train_data/noisy_test_{0}_{1}.pkl".format(e, data_version), "rb") as f:
    noisy_test = pickle.load(f)

def mse(pi, dataset):
    total = 0
    s_traj = [ds[0] for ds in dataset]
    a_traj = [ds[1] for ds in dataset]

    for i in range(len(s_traj)):
        total += np.linalg.norm(pi(s_traj[i]) - a_traj[i])
    return total / len(s_traj)

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
start_time = time.time()
for i in range(0, 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    for size in [50]:
        with open("./data/{0}_train_data/train_{1}_{2}_{0}_{3}.pkl".format(e, size, i, data_version),"rb") as f :
            expert_trajs = pickle.load(f)
        if AIT:
            hop = TCN_test(expert_trajs_data=expert_trajs)
            print(hop)
        else:
            hop = 1
        start_time_bc = time.time()
        pi_BC = BC(expert_trajs, Model(env.observation_space.shape[0], env.action_space.shape[0]).to(device), device=device).to(device)
        bc_mse_noisy_i.append(mse(lambda s: pi_BC(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), noisy_test))
        bc_mse_denoised_i.append(mse(lambda s: pi_BC(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), denoised_test))
        bc_J_noisy_i.append(eval_policy(lambda s: pi_BC(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), env, noisy=True, step= 100, TCN = TCN))
        bc_J_denoised_i.append(eval_policy(lambda s: pi_BC(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), env, noisy=False))
        print('BC',
                size,
                bc_mse_noisy_i[-1],
                bc_mse_denoised_i[-1],
                bc_J_noisy_i[-1], 
                bc_J_denoised_i[-1])
        start_time_ccil = time.time()
        print("trajsize 50 Time used for BC:", start_time_ccil - start_time_bc)
        pi_CCIL = CCIL(expert_trajs,
                    Model(env.observation_space.shape[0], env.action_space.shape[0]).to(device),
                    lambda a, b: dynamics(a, b, env), pi_BC=pi_BC, nsamp=4,  step = 100, sigma = 5, hop = hop, device = device).to(device)   
        CCIL_mse_noisy_i.append(mse(lambda s: pi_CCIL(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), noisy_test))
        CCIL_mse_denoised_i.append(mse(lambda s: pi_CCIL(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), denoised_test))
        CCIL_J_noisy_i.append(eval_policy(lambda s: pi_CCIL(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), env, noisy=True, step= 100, TCN = TCN))
        CCIL_J_denoised_i.append(eval_policy(lambda s: pi_CCIL(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), env, noisy=False))
        print('CCIL',
                size,
                CCIL_mse_noisy_i[-1],
                CCIL_mse_denoised_i[-1],
                CCIL_J_noisy_i[-1],
                CCIL_J_denoised_i[-1])
        end_time_ccil = time.time()
        print("trajsize 50 Time used for CCIL:", end_time_ccil - start_time_ccil)
        pi_CCIL2 = CCIL2(expert_trajs,
                        Model(env.observation_space.shape[0], env.action_space.shape[0]).to(device),
                        Model(env.observation_space.shape[0], env.action_space.shape[0]).to(device),hop = hop, device = device).to(device)
        CCIL2_mse_noisy_i.append(mse(lambda s: pi_CCIL(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), noisy_test))
        CCIL2_mse_denoised_i.append(mse(lambda s: pi_CCIL(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), denoised_test))
        CCIL2_J_noisy_i.append(eval_policy(lambda s: pi_CCIL(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), env, noisy=True, step= 100, TCN = TCN))
        CCIL2_J_denoised_i.append(eval_policy(lambda s: pi_CCIL(torch.from_numpy(s).float().to(device)).detach().cpu().numpy(), env, noisy=False))
        print('CCIL2',
                size,
                CCIL2_mse_noisy_i[-1],
                CCIL2_mse_denoised_i[-1],
                CCIL2_J_noisy_i[-1],
                CCIL2_J_denoised_i[-1])
        end_time_ccil2 = time.time()
        print("trajsize 50 Time used for CCIL2:", end_time_ccil2 - end_time_ccil)
    # bc_mse_noisy.append(bc_mse_noisy_i)
    # bc_mse_denoised.append(bc_mse_denoised_i)
    # bc_J_noisy.append(bc_J_noisy_i)
    # bc_J_denoised.append(bc_J_denoised_i)

    # with open("data/output_mse&J/{0}/bc_mse_noisy_{0}_{1}.pkl".format(e, method),"wb") as f:
    #     pickle.dump(bc_mse_noisy,f)
    # with open("data/output_mse&J/{0}/bc_mse_denoised_{0}_{1}.pkl".format(e, method),"wb") as f:
    #     pickle.dump(bc_mse_denoised,f)
    # with open("data/output_mse&J/{0}/bc_J_noisy_{0}_{1}.pkl".format(e, method),"wb") as f:
    #     pickle.dump(bc_J_noisy,f)
    # with open("data/output_mse&J/{0}/bc_J_denoised_{0}_{1}.pkl".format(e, method),"wb") as f:
    #     pickle.dump(bc_J_denoised,f)

    
    # CCIL_mse_noisy.append(CCIL_mse_noisy_i)
    # CCIL_mse_denoised.append(CCIL_mse_denoised_i)
    # CCIL_J_noisy.append(CCIL_J_noisy_i)
    # CCIL_J_denoised.append(CCIL_J_denoised_i)
    
    # with open("data/output_mse&J/{0}/CCIL_mse_noisy_{0}_{1}.pkl".format(e, method),"wb") as f :
    #     pickle.dump(CCIL_mse_noisy,f)
    # with open("data/output_mse&J/{0}/CCIL_mse_denoised_{0}_{1}.pkl".format(e, method),"wb") as f :
    #     pickle.dump(CCIL_mse_denoised,f)
    # with open("data/output_mse&J/{0}/CCIL_J_noisy_{0}_{1}.pkl".format(e, method),"wb") as f :
    #     pickle.dump(CCIL_J_noisy,f)
    # with open("data/output_mse&J/{0}/CCIL_J_denoised_{0}_{1}.pkl".format(e, method),"wb") as f :
    #     pickle.dump(CCIL_J_denoised,f)

    
    # CCIL2_mse_noisy.append(CCIL2_mse_noisy_i)
    # CCIL2_mse_denoised.append(CCIL2_mse_denoised_i)
    # CCIL2_J_noisy.append(CCIL2_J_noisy_i)
    # CCIL2_J_denoised.append(CCIL2_J_denoised_i)

    # with open("data/output_mse&J/{0}/CCIL2_mse_noisy_{0}_{1}.pkl".format(e, method),"wb") as f :
    #     pickle.dump(CCIL2_mse_noisy,f)
    # with open("data/output_mse&J/{0}/CCIL2_mse_denoised_{0}_{1}.pkl".format(e, method),"wb") as f :
    #     pickle.dump(CCIL2_mse_denoised,f)
    # with open("data/output_mse&J/{0}/CCIL2_J_noisy_{0}_{1}.pkl".format(e, method),"wb") as f :
    #     pickle.dump(CCIL2_J_noisy,f)
    # with open("data/output_mse&J/{0}/CCIL2_J_denoised_{0}_{1}.pkl".format(e, method),"wb") as f :
    #     pickle.dump(CCIL2_J_denoised,f)
    end_time = time.time()
    print("Time used:", end_time - start_time)
              