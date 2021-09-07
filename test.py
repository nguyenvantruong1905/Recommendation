
import pandas as pd
import numpy as np
import pickle

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
# predict=[]
def predict(user,qi_a,pu_a,qi,pu,k):
    predict = []
    for item in range(qi.shape[0]):
        pre = (
                global_mean_
                + pu[user].T @ (qi_a)
                + qi[item].T @ (pu_a)
                + qi[item].T @ pu[user]
            )
        predict.append(pre)
    pred_numpy = np.array(predict)
    pred_maps = np.argsort(pred_numpy)
    top_k = []
    for i in range(k):
        c = len(qi)-i-1
        top_k.extend([k for k,v in item_map.items() if v==pred_maps[c]])
    return top_k
def recall_top_k(user,qi_a,pu_a,qi,pu,k):
    recall = []
    for u in user:
        top_k = predict(u,qi_a,pu_a,qi,pu,k)
        val_a= val.loc[val["u_id"] ==u]
        c=0
        for i in top_k:
            val_a= val_a.loc[val_a["i_id"] == i]
            c+=len(val_a.index)
        recall.append(c/10)
    return sum(recall)/len(recall)
user  = val["u_id"].unique().tolist()
print(recall_top_k(user,qi_a,pu_a,qi,pu,10))
    # a["predict"]=predict(u,qi_a,pu_a,qi,pu,5)
    # print(a)
#     # val['rating'] = np.where(val['u_id']==u)
#     # # a = a.sort_values(by='predict', ascending=False)
#     # print(val["rating"])
#     # print(a,b,c,d,e)
