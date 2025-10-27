import pandas as pd
import os,random
print(os.path)
os.environ['R_HOME'] = 'D:\ApplicationSpace\Anaconda3\envs\py38\Lib\R'

from src.AIT_condition import AIT_test
# from rpy2.robjects import r
from pandas.core.frame import DataFrame

# import pickle

def AIT_condition(data = None):

    Z = 'IV_candidate'
    A_Z = AIT_test(data, Z, relation='linear' )   # relation = 'linear' or 'nonlinear'
    # if A_Z['IV_validity']: print(f'The candidate State is valid IV!!!')
    # else: print(f'The candidate State is invalid IV!')
    return A_Z['IV_validity'],A_Z['pValue_Z']


def TCN_test(expert_trajs_data = None): 
    # index = []
    rand_index = random.randint(1, 21)
    index_list = []
    for traj in expert_trajs_data:
        p_value_list = []
        IV_list = []

        for j in range(20, 0, -1):

            state, action, candidate_state = [], [], []
            state.append(traj[0][j:])
            action.append(traj[1][j:])
            candidate_state.append(traj[0][:len(traj[0])-j])
            data_dict = {"Treatment": state, "Outcome": action, "IV_candidate": candidate_state}
            AIT_data = DataFrame(data_dict,columns=["Treatment", "Outcome", "IV_candidate"])
            is_IV, p_value = AIT_condition(data = AIT_data)
            p_value_list.append(p_value) 
            IV_list.append(is_IV) 
        
        candidates = [i for i in range(len(IV_list)-1) if IV_list[i] and not IV_list[i+1]]
        if candidates:
            index_list.append(20-max(candidates))
    if index_list:
        result = max(set(index_list), key=index_list.count)
        max_count = index_list.count(result)
        candidates = [i for i in set(index_list) if index_list.count(i) == max_count]
        result = max(candidates)
    else:
        result = rand_index       
    return result

# calculate accuracy of valid IV indentification
def IV_index_List_identify(expert_trajs_data = None): 
    # index = []
    rand_index = random.randint(1, 21)
    index_list = []
    for traj in expert_trajs_data:
        p_value_list = []
        IV_list = []

        for j in range(20, 0, -1):

            state, action, candidate_state = [], [], []
            state.append(traj[0][j:])
            action.append(traj[1][j:])
            candidate_state.append(traj[0][:len(traj[0])-j])
            data_dict = {"Treatment": state, "Outcome": action, "IV_candidate": candidate_state}
            AIT_data = DataFrame(data_dict,columns=["Treatment", "Outcome", "IV_candidate"])
            is_IV, p_value = AIT_condition(data = AIT_data)
            p_value_list.append(p_value)
            IV_list.append(is_IV) 
      
        candidates = [i for i in range(len(IV_list)-1) if IV_list[i] and not IV_list[i+1]]

        if candidates:
            index_list.append(20-max(candidates))
        else:
            index_list.append(rand_index)
                   
    return index_list


def acc_calculate(data = None, pace = 1):
    if len(data) == 0:
        return 0
    accuracy = len([i for i in data if i > pace])/len(data) 
    return accuracy

def max_index(inx_lst):
    index = []
    max_n = max(inx_lst)
    for i in range(len(inx_lst)):
        if inx_lst[i] == max_n:
            index.append(i)
    return index # list