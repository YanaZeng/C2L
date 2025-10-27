import numpy as np
# import sys
# sys.path.insert(0, './src')
from src.lunar_lander_env import *
# from lunar_lander_env import *

def get_state(env):
    pos = env.lander.position
    vel = env.lander.linearVelocity
    state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - (env.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            env.lander.angle,
            20.0*env.lander.angularVelocity/FPS,
    ]
    return np.array(state)

def set_state(env, s):
    env.lander.position.x = (s[0] * (VIEWPORT_W/SCALE/2)) + VIEWPORT_W/SCALE/2
    env.lander.position.y = (s[1] * (VIEWPORT_H/SCALE/2)) + (env.helipad_y+LEG_DOWN/SCALE)
    env.lander.linearVelocity.x = s[2] / ((VIEWPORT_W/SCALE/2)/FPS)
    env.lander.linearVelocity.y = s[3] / ((VIEWPORT_H/SCALE/2)/FPS)
    env.lander.angle = s[4] * 1.0
    env.lander.angularVelocity = s[5] * FPS / 20.0

def T(s, a, sim_env):
    sim_env.reset() 
    set_state(sim_env, s)
    obs, _, _, _ = sim_env.step(a)
    return obs

def dynamics(S, A, sim_env):
    S_prime = []
    for (s, a) in zip(S, A):
        s_prime = T(s, a, sim_env)
        S_prime.append(s_prime)
    return np.stack(S_prime, axis=0)

def rollout(pi, env):
    states = []
    actions = []
    s = env.reset()
    done = False
    J = 0
    t = 0
    gamma = 0.9
    while not done:
        t += 1
        states.append(s.reshape(-1))
        a = pi(s)
        if isinstance(a, tuple):
            a = a[0]
        actions.append(a.reshape(-1))
        s, r, done, _ = env.step(a)
        J += r
    states = np.array(states, dtype='float')
    actions = np.array(actions, dtype='float')
    return states, actions, J

def noisy_rollout(pi, env, sigma=0.5, step = 0, u_len = 3, hop = 3, distribution = 'normal' , TCN = False):
    states = []
    actions = []
    s = env.reset()
    done = False
    u_past = 0
    u_past_1, u_past_2, u_past_3, u_past_4, u_past_5, u_past_6 = 0, 0, 0, 0, 0, 0
    J = 0
    if TCN:
        index = np.random.randint(0,100)
    else:
        index = -1
    i = 0
    while not done:
        i += 1
        states.append(s.reshape(-1))
        a = pi(s)
        if isinstance(a, tuple):
            a = a[0]
        if not TCN:
            u = np.random.normal() * sigma
            a = a + 0.1 * u + 8.0 * u_past
            u_past = u
        else:
            while i in range(index,index+step):
                if distribution == 'normal':
                    u = np.random.normal() * sigma
                elif distribution == 'uniform':
                    u = np.random.uniform(0,1) * sigma
                elif distribution == 'exponential':
                    u = np.random.exponential(1.5) * sigma
                elif distribution == 'gamma':
                    u = np.random.gamma(0.5,0.1) * sigma
                elif distribution == 'beta':
                    u = np.random.beta(5,1) * sigma
                
                elif distribution == 'laplace':
                    u = np.random.laplace(4,2) * sigma
                elif distribution == 'lognormal':
                    u = np.random.lognormal(0,1) * sigma
                elif distribution == 't':
                    u = np.random.standard_t(2) * sigma
                
                    
                if hop == 3:
                    a = a * u * u_past_1 * u_past_2 * 0.1
                    u_past_1, u_past_2 = u, u_past_1
                elif hop == 4:
                    a = a * u * u_past_1 * u_past_2 * u_past_3 * 0.1
                    u_past_1, u_past_2, u_past_3 = u, u_past_1, u_past_2
                elif hop == 5:
                    a = a * u * u_past_1 * u_past_2 * u_past_3 * u_past_4 * 0.1
                    u_past_1, u_past_2, u_past_3, u_past_4 = u, u_past_1 , u_past_2, u_past_3
                elif hop == 6:
                    a = a * u * u_past_1 * u_past_2 * u_past_3 * u_past_4 * u_past_5 * 0.1
                    u_past_1, u_past_2, u_past_3, u_past_4, u_past_5 = u, u_past_1 , u_past_2, u_past_3, u_past_4
                elif hop == 7:
                    a = a * u * u_past_1 * u_past_2 * u_past_3 * u_past_4 * u_past_5 * u_past_6 * 0.1
                    u_past_1, u_past_2, u_past_3, u_past_4, u_past_5, u_past_6 = u, u_past_1 , u_past_2, u_past_3, u_past_4, u_past_5
                break
        
        actions.append(a.reshape(-1))
        s, r, done, _ = env.step(a, index, step, hop, distribution)
        J += r
    states = np.array(states, dtype='float')
    actions = np.array(actions, dtype='float')
    return states, actions, J

def eval_policy(pi, env, noisy=False,step = 100, TCN = False):
    Js = []
    for _ in range(100):
        if noisy:
            s_traj, a_traj, J = noisy_rollout(pi, env, step= step, TCN=TCN)
        else:
            s_traj, a_traj, J = rollout(pi, env)
        Js.append(J)
    return np.mean(Js)
