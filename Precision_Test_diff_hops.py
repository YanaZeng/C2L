from src.TCN_AIT_test import IV_index_List_identify, acc_calculate
import pickle
import numpy as np
import random
env = 'll'
if __name__ == '__main__':
    acc_list, avg_cc = [], {}
    pace = 1
    hops = [3,4,5,6,7]
    for hop in hops:
        data_version = str(hop) + '_TCN'
        for i in range(0 , 5):
            accuracy = []
            for size in [10, 20, 30, 40, 50]:
                print("hop = ",hop, "i = ", i, "size = ", size)
                with open("data/{0}_diff_hop/train_{1}_{2}_{0}_{3}.pkl".format(env, size, i, data_version),"rb") as f :
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

                index = IV_index_List_identify(expert_trajs_data=expert_trajs_data)
                # print("index = ", index)
                accuracy.append(acc_calculate(index, pace=pace))

            acc_list.append(accuracy)
        pace += 1
        avg_size = np.array(acc_list).mean(axis = 0)
        avg_cc[str(hop)] = avg_size
    print(avg_cc)
    with open("data/output_accuracy/{0}_Accuracy&Diff_Hops.pkl".format(env), "wb") as f:
        pickle.dump(avg_cc, f)