import pickle
import pandas as pb
import numpy as np
from datetime import datetime
from numba import jit
params = open(r"C:\Users\nvtru\Desktop\AI-Recommender\params",'rb')
user_item_map = open(r"C:\Users\nvtru\Desktop\AI-Recommender\user_item_map",'rb')
params = pickle.load(params)
user_item_map = pickle.load(user_item_map)

user_map = user_item_map['user_map']
item_map = user_item_map['item_map']
pu = params['pu']
qi = params["qi"]
pu_a = params["pu_a"]
qi_a = params["qi_a"]
tu = params["tu"]
lambda1 =params["lambda1"]
lambda2 = params['lambda2']
global_mean = params['global_mean']
alpha=params['alpha']
beta = params['beta']
def time_days(tu):
    dt = datetime.utcnow()
    dt64 = np.datetime64(dt)
    ts = abs(dt64 - tu)
    return(ts.astype('timedelta64[D]')/ np.timedelta64(1, 'D'))
np.array(tu)
time_edulive =alpha*(np.log(1+(time_days(tu))**beta))
print(item_map[1198])
def predict(u_id,pu, qi, pu_a, qi_a, global_mean,lambda1,lambda2, recommend = 'higher'):
    u_id = user_map[u_id]
    arr =[]
    for item in range(qi.shape[0]):
        pred = (
        global_mean
        + (pu[u_id] + time_edulive[u_id]) .T @ (qi_a)
        + qi[item].T @ (pu_a)
        + qi[item].T @ (pu[u_id] +  time_edulive[u_id])
        - lambda1*33
        - lambda2*20
        )
        arr.append(pred)
    arr_higher = np.array(arr)
    arr_improve = 5-arr_higher
    if recommend == 'higher':
        arr_map = np.argsort(arr_higher)
        recommend_item1 = item_map[arr_map[0]]
        recommend_item2 = item_map[arr_map[1]]
    if recommend == 'improve':
        arr_map = np.argsort(arr_improve)
        print(arr_map)
        recommend_item1 = item_map[arr_map[0]]
        recommend_item2 = item_map[arr_map[1]]
    return recommend_item1, recommend_item2
print(predict(278,pu, qi, pu_a, qi_a, global_mean,lambda1,lambda2, recommend='improve'))