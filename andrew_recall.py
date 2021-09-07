
import numpy as np
import pickle
import pandas as pd
from numba import jit
import os
path_file_train = (r"C:\Users\nvtru\Desktop\auto-recommender\movieslen_data\train.txt")
path_file_val = (r"C:\Users\nvtru\Desktop\auto-recommender\movieslen_data\test1.txt")
train = pd.read_csv(path_file_train, sep=",",
                   names=["u_id", "i_id", "rating", "timestamps"])
val = pd.read_csv(path_file_val, sep=",",
                   names=["u_id", "i_id", "rating", "timestamps"])

def _preprocess_train(X):
    user_ids = X["u_id"].unique().tolist()
    item_ids = X["i_id"].unique().tolist()
    n_users = len(user_ids)
    n_items = len(item_ids)
    user_idx = range(n_users)
    item_idx = range(n_items)
    user_mapping_ = dict(zip(user_ids, user_idx))
    item_mapping_ = dict(zip(item_ids, item_idx))
    return user_mapping_,item_mapping_

def _preprocess_val(X,user_mapping_,item_mapping_):
    X["u_id"] = X["u_id"].map(user_mapping_)
    X["i_id"] = X["i_id"].map(item_mapping_)
    X.fillna(-1, inplace=True)
    X["u_id"] = X["u_id"].astype(np.int32)
    X["i_id"] = X["i_id"].astype(np.int32)
    return X[["u_id", "i_id", "rating"]].values
user_map, item_map = _preprocess_train(train)
X=_preprocess_val(val,user_map,item_map)
path_param = r"C:\Users\nvtru\Desktop\auto-recommender\resuld\movielens_edulive_params"
params = pickle.load(open(path_param,"rb"))
pu = params["pu"]
qi = params["qi"]
pu_a = params["pu_a"]
qi_a = params["qi_a"]
global_mean_=params["global_mean_"]
user  = np.unique(X[:,0])

@jit
def recall_top_k_user(pre,user,qi,k,):
    recall = []
    n_shape =qi.shape[1]
    for i in range(n_shape):
        pred = pre[:,-(i+1):]
        sum_pred = np.sum(pred,axis=1)
        pred_maps = np.argsort(sum_pred)
        top_k = []
        for i in range(k):
            top_k.append(pred_maps[(k-1-i)])
        X_user = X[np.where(X[:,0]==user)]
        c=0
        for i in top_k:
            X_user_top = X_user[np.where((X_user[:,1]==i)&(X_user[:,2]==5))]
            c+=len(X_user_top)
        recall.append((c/k))
    return np.array(recall)
def recall_mean_all_user_feature(qi,pu,k,):
    X=np.zeros(qi.shape[1])
    c=0
    for u in user:
        pre = pu[u]*qi
        pre = np.sort(pre,axis=1)
        recall = recall_top_k_user(pre,u,qi,k,)
        recall = np.array(recall)
        X=X+recall
        c=c+1
        print("user da xu ly",c,"/",len(user))
    return X/len(user)
print(recall_mean_all_user_feature(qi,pu,10,))

