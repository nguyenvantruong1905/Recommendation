import pandas as pd
import numpy as np
import pickle

# path_file_train = (r"C:\Users\nvtru\Desktop\auto-recommender\movieslen_data\train.txt")
# path_file_test = (r"C:\Users\nvtru\Desktop\auto-recommender\movieslen_data\test1.txt")
# train = pd.read_csv(path_file_train, sep=",",
#                 names=["u_id", "i_id", "rating", "timestamps"])
# test = pd.read_csv(path_file_test, sep=",",
#                 names=["u_id", "i_id", "rating", "timestamps"])
# time_exam_train = np.random.choice(30, 80000)
# time_exam_train = np.random.choice(30, 80000)
# time_exam_test = np.random.choice(30, 20000)
# action_exam_train = np.random.choice(50, 80000)
# action_exam_test = np.random.choice(50, 20000)
# train['time_exam'] = pd.Series(time_exam_train,index=train.index)
# train['action_exam'] = pd.Series(action_exam_train,index=train.index)
# test['time_exam'] = pd.Series(time_exam_test,index=test.index)
# test['action_exam'] = pd.Series(action_exam_test,index=test.index)

def map_data(train,filename):
    user_ids = train["u_id"].unique().tolist()
    item_ids = train["i_id"].unique().tolist()
    n_users = len(user_ids)
    n_items = len(item_ids)
    user_idx = range(n_users)
    item_idx = range(n_items)

    user_mapping = dict(zip(user_ids, user_idx))
    item_mapping = dict(zip(item_ids, item_idx))
    data = {
        "user_map": user_mapping,
        "item_map": item_mapping
    }
    dbfile = open(filename+"user_item_map", "wb")
    pickle.dump(data, dbfile)
    return user_mapping, item_mapping
def pre_process_data(train,test,user_mapping,item_mapping):
    X=train.copy()
    X_test=test.copy()
    X["u_id"] = X["u_id"].map(user_mapping)
    X["i_id"] = X["i_id"].map(item_mapping)
    X["u_id"] = X["u_id"].astype(np.int32)
    X["i_id"] = X["i_id"].astype(np.int32)  
    X_test["u_id"] = X_test["u_id"].map(user_mapping)
    X_test["i_id"] = X_test["i_id"].map(item_mapping)
    X_test.fillna(-1, inplace=True)
    X_test["u_id"] = X_test["u_id"].astype(np.int32)
    X_test["i_id"] = X_test["i_id"].astype(np.int32)  
    
    return X,X_test
# filename = r"C:/Users/nvtru/Desktop/auto-recommender/edulive/abc"
# user_mapping, item_mapping = map_data(train,filename)
# print(pre_process_data(train,test,user_mapping, item_mapping))