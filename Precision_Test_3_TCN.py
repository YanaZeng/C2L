import os
print(os.path)
os.environ['R_HOME'] = 'D:\ApplicationSpace\Anaconda3\envs\py38\Lib\R'

from src.TCN_AIT_test import IV_index_List_identify, acc_calculate

import pickle
import numpy as np

env = 'll'
if __name__ == '__main__':
    acc_list = []
    data_version = '3_TCN'
    pace = 1
    for i in range(0,5):  
        accuracy = []
        for size in [10, 20, 30, 40, 50]:
            print("i = ", i, "size = ", size)
            with open("./data/{0}_train_data/train_{1}_{2}_{0}_{3}.pkl".format(env, size, i, data_version),"rb") as f :
                expert_trajs = pickle.load(f)

            if env == 'll':
                expert_trajs_data = expert_trajs
            elif env == 'ant':
                s_trajs = expert_trajs["s"]
                a_trajs = expert_trajs["a"]
                p_trajs = expert_trajs["P"]
                v_trajs = expert_trajs["V"]
                c_trajs = expert_trajs["C"]
                expert_trajs_data = [[s_trajs[i], a_trajs[i], p_trajs[i], v_trajs[i], c_trajs[i]] for i in range(len(s_trajs))]
            elif env == 'hc':
                s_trajs = expert_trajs["s"]
                a_trajs = expert_trajs["a"]
                p_trajs = expert_trajs["P"]
                v_trajs = expert_trajs["V"]
                expert_trajs_data = [[s_trajs[i], a_trajs[i], p_trajs[i], v_trajs[i]] for i in range(len(s_trajs))]
            
            index = IV_index_List_identify(expert_trajs_data)
            print("index = ", index)
            accuracy.append(acc_calculate(index, pace = pace)) # pace在3-TCN中为1，4-TCN中为2等等，只有index值大于pace才算识别正确

        acc_list.append(accuracy)
    
    print(acc_list)
    print(np.array(acc_list).mean(axis = 0))
    with open("data/output_accuracy/{0}_Accuracy&Num_traj.pkl".format(env), "wb") as f :
        pickle.dump(acc_list,f)
