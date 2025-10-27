import gym
import numpy as np
import tqdm
import os
os.add_dll_directory("C://Users//dell//.mujoco//mjpro150//bin")
os.add_dll_directory("C://Users//dell//.mujoco//mujoco-py-1.50.1.0//mujoco_py")
from mujoco_py import MjSimState


class CnfndWrapper(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, base_env, sigma=0.5):
        super(CnfndWrapper, self).__init__()
        self.base_env = base_env 
        self.iter = 0 
        self.observation_space = self.base_env.observation_space 
        self.action_space = self.base_env.action_space
        self.sigma = sigma
        self.u_past_1 = np.random.normal() * self.sigma
        self.u_past_2 = np.random.normal() * self.sigma
    def step(self, action, index = 0 , n = 1000, distribution = 'normal', hop = 3):
        if index > 0 and self.iter in range(index, index + n):
            if distribution == 'normal':
                u = np.random.normal() * self.sigma
            elif distribution == 'uniform':
                u = np.random.uniform(0, 1) * self.sigma
            elif distribution == 'gamma':
                u = np.random.gamma(0.5, 0.1) * self.sigma
            elif distribution == 'exponential':
                u = np.random.exponential(1.5) * self.sigma
            elif distribution == 'lognormal':
                u = np.random.lognormal(0, 1) * self.sigma
            elif distribution == 'beta':
                u = np.random.beta(5, 1) * self.sigma
            elif distribution == 'laplace':
                u = np.random.laplace(4, 2) * self.sigma
            elif distribution == 't':
                u = np.random.standard_t(2) * self.sigma

            if hop == 3:    
                action = action * u * self.u_past_1 * self.u_past_2
                self.u_past_1, self.u_past_2 = u, self.u_past_1
            elif hop == 4:
                action = action * u * self.u_past_1 * self.u_past_2 * self.u_past_3
                self.u_past_1, self.u_past_2, self.u_past_3 = u, self.u_past_1, self.u_past_2
            elif hop == 5:
                action = action * u * self.u_past_1 * self.u_past_2 * self.u_past_3 * self.u_past_4
                self.u_past_1, self.u_past_2, self.u_past_3, self.u_past_4 = u, self.u_past_1, self.u_past_2, self.u_past_3
            elif hop == 6:
                action = action * u * self.u_past_1 * self.u_past_2 * self.u_past_3 * self.u_past_4 * self.u_past_5
                self.u_past_1, self.u_past_2, self.u_past_3, self.u_past_4, self.u_past_5 = u, self.u_past_1, self.u_past_2, self.u_past_3, self.u_past_4
            elif hop == 7:
                action = action * u * self.u_past_1 * self.u_past_2 * self.u_past_3 * self.u_past_4 * self.u_past_5 * self.u_past_6
                self.u_past_1, self.u_past_2, self.u_past_3, self.u_past_4, self.u_past_5, self.u_past_6 = u, self.u_past_1, self.u_past_2, self.u_past_3, self.u_past_4, self.u_past_5
        else:
            u = np.random.normal() * self.sigma
            action = action + 1 * self.u_past + 1 * u
            self.u_past = u
        next_obs, reward, done, info = self.base_env.step(action)
        return next_obs, reward, done, info
    def reset(self):
        self.u_past_1 = np.random.normal() * self.sigma
        self.u_past_2 = np.random.normal() * self.sigma
        self.u_past_3 = np.random.normal() * self.sigma
        self.u_past_4 = np.random.normal() * self.sigma
        self.u_past_5 = np.random.normal() * self.sigma
        self.u_past_6 = np.random.normal() * self.sigma

        self.u_past = np.random.normal() * self.sigma
        obs = self.base_env.reset()
        return obs
    def render(self, mode='human'): 
        self.base_env.render(mode=mode)
    def close (self):
        self.base_env.close()

def get_state(env):
    sim_state = env.base_env.env.sim.get_state()
    qpos = sim_state.qpos.copy() #position
    qvel = sim_state.qvel.copy() #veloctiy
    return qpos, qvel

def set_state(env, qpos, qvel):
    sim = env.base_env.sim
    state = sim.get_state()

    new_state = MjSimState(
        time = state.time,
        qpos = qpos,
        qvel = qvel,
        act = state.act,
        udd_state = state.udd_state
    )
    sim.set_state(new_state)
    sim.forward() 

def T(qpos, qvel, a, a_e, sim_env):
    sim_env.reset() 
    set_state(sim_env, qpos, qvel) 
    for t in range(len(a_e)):
        sim_env.step(a_e[t])
    obs, _, _, _ = sim_env.step(a)
    return obs 

def dynamics(P, V, A, A_exp, sim_env):
    S_prime = []
    for t in tqdm.tqdm(range(len(A))):
        a = A[t] 
        a_e = A_exp[:t]
        s_prime = T(P[0], V[0], a, a_e, sim_env) 
        S_prime.append(s_prime)
    return np.stack(S_prime, axis=0)

def rollout(pi, env, full_state=False):
    states = []
    actions = []
    if full_state: 
        qpos = []
        qvel = []
    s = env.reset() 
    if full_state:
        p, v= get_state(env)
        qpos.append(p)
        qvel.append(v)
    done = False
    J = 0
    while not done: 
        states.append(s.reshape(-1))
        a = pi(s)
        if isinstance(a, tuple):
            a = a[0]
        actions.append(a.reshape(-1))
        s, r, done, _ = env.step(a)
        if full_state: 
            p, v= get_state(env)   
            qpos.append(p)
            qvel.append(v)
        J += r
    states = np.array(states, dtype='float')
    actions = np.array(actions, dtype='float')
    if full_state:
        return states, actions, J, qpos, qvel
    else:
        return states, actions, J

def noisy_rollout(pi, env, sigma=3, full_state=False,n = 0, distribution = 'normal', hop = 3, gamma=0.6, TCN = False):
    states = []
    actions = []
    if full_state:
        qpos = []
        qvel = []
    s = env.reset()
    if full_state:
        p, v = get_state(env)
        qpos.append(p)
        qvel.append(v)
    done = False
    u_past = 0
    u_past_1, u_past_2,u_past_3,u_past_4,u_past_5,u_past_6 = 0, 0, 0, 0, 0, 0
    J = 0
    i = 0
    if TCN:
        index = np.random.randint(0,500)
    else:
        index = -1 
    while not done:
        i += 1
        states.append(s.reshape(-1))
        a = pi(s)
        if isinstance(a, tuple):
            a = a[0]
        if index == -1: 
            u = np.random.normal() * sigma 
            a = a + 1 * u + 1.0 * u_past 
            u_past = u 

        else:
            while i in range(index, index + n):
                if distribution == 'normal':
                    u = np.random.normal() * sigma
                elif distribution == 'uniform':
                    u = np.random.uniform(0, 1) * sigma
                elif distribution == 'gamma':
                    u = np.random.gamma(0.5, 0.1) * sigma
                elif distribution == 'exponential':
                    u = np.random.exponential(1.5) * sigma
                elif distribution == 'lognormal':
                    u = np.random.lognormal(0, 1) * sigma
                elif distribution == 'beta':
                    u = np.random.beta(5, 1) * sigma
                elif distribution == 'laplace':
                    u = np.random.laplace(4, 2) * sigma
                elif distribution == 't':
                    u = np.random.standard_t(2) * sigma
                
                if hop == 3:
                    a = a * u * u_past_1 * u_past_2
                    u_past_1, u_past_2 = u, u_past_1
                elif hop == 4:
                    a = a * u * u_past_1 * u_past_2 * u_past_3
                    u_past_1, u_past_2, u_past_3 = u, u_past_1, u_past_2
                elif hop == 5:
                    a = a * u * u_past_1 * u_past_2 * u_past_3 * u_past_4
                    u_past_1, u_past_2, u_past_3, u_past_4 = u, u_past_1, u_past_2, u_past_3
                elif hop == 6:
                    a = a * u * u_past_1 * u_past_2 * u_past_3 * u_past_4 * u_past_5
                    u_past_1, u_past_2, u_past_3, u_past_4, u_past_5 = u, u_past_1, u_past_2, u_past_3, u_past_4
                elif hop == 7:
                    a = a * u * u_past_1 * u_past_2 * u_past_3 * u_past_4 * u_past_5 * u_past_6
                    u_past_1, u_past_2, u_past_3, u_past_4, u_past_5, u_past_6 = u, u_past_1, u_past_2, u_past_3, u_past_4, u_past_5
                break

        actions.append(a.reshape(-1))
        s, r, done, _ = env.step(a, index, n, distribution, hop)
        if full_state:
            p, v = get_state(env)
            qpos.append(p)
            qvel.append(v)
        J += r
    states = np.array(states, dtype='float')
    actions = np.array(actions, dtype='float')
    if full_state:
        return states, actions, J, qpos, qvel
    else:
        return states, actions, J
