
import numpy as np
import pickle
import pandas as pd
from numba import jit
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
# @jit
def recall_top_k_user(user,qi,pu,pu_a,qi_a,k,global_mean_):
    # pre = pu[user]*qi
    pre = (
            global_mean_
            + pu[user]*(qi_a)
            + qi*(pu_a)
            + qi*(pu[user])

        )
    # print(pre)
    pre = np.sort(pre,axis=1)
    recall = []
    for i in range(qi.shape[1]):
        c=-(i+1)
        pred = pre[:,c:]
        sum_pred = np.sum(pred,axis=1)
        pred_maps = np.argsort(sum_pred)
        top_k = []
        for i in range(k):
            c = len(pred_maps)-(i+1)
            top_k.extend([k for k,v in item_map.items() if v==pred_maps[c]])
        val_user= val.loc[val["u_id"] ==user]
        c=0
        for i in top_k:
            val_user_top= val_user.loc[(val_user["i_id"] == i) & (val_user["rating"] == 5)]
            # val_user_top_rate5 = val_user_top.loc[val_user_top["rating"]==5]
            c+=len(val_user_top.index)
        recall.append((c/k))
    return recall
# recall_top_k_user(178,qi_a,pu_a,qi,pu,10)  
user  = val["u_id"].unique().tolist()
# @jit
def recall_mean_all_user_feature(qi,pu,pu_a,qi_a,k,global_mean_):
    X=np.zeros(qi.shape[1])
    print(X)
    for u in user:
        recall = recall_top_k_user(u,qi,pu,pu_a,qi_a,10,global_mean_)
        recall = np.array(recall)
        X=X+recall
    return X/len(user)
print(recall_mean_all_user_feature(qi,pu,pu_a,qi_a,10,global_mean_))
# print(recall_top_k_user(198,qi,pu,pu_a,qi_a,10,global_mean_))
# print("%0.6f" %X[0],"%0.6f" %X[1],"%0.6f" %X[2])
# print("%0.6f" %(X/len(user))[0],"%0.6f" %(X/len(user))[1],"%0.6f" %(X/len(user))[2])
