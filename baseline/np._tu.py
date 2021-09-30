import numpy as np
from numpy.core.fromnumeric import argsort
import pandas as pd
from datetime import datetime
import math
path_file_train = (r"C:\Users\nvtru\Desktop\AI-Recommender\movieslen_data\try.txt")
path_file_test = (r"C:\Users\nvtru\Desktop\AI-Recommender\movieslen_data\testttt.txt")
train = pd.read_csv(path_file_train, sep=",",
                   names=["u_id", "i_id", "rating", "timestamps"])
test = pd.read_csv(path_file_test, sep=",",
                   names=["u_id", "i_id", "rating", "timestamps"])
def _preprocess_data(X):

    user_ids = X["u_id"].unique().tolist()
    item_ids = X["i_id"].unique().tolist()
    n_users = len(user_ids)
    n_items = len(item_ids)
    user_idx = range(n_users)
    item_idx = range(n_items)

    user_mapping_ = dict(zip(user_ids, user_idx))
    item_mapping_ = dict(zip(item_ids, item_idx))
    X["u_id"] = X["u_id"].map(user_mapping_)
    X["i_id"] = X["i_id"].map(item_mapping_)

    X.fillna(-1, inplace=True)
    X["u_id"] = X["u_id"].astype(np.int32)
    X["i_id"] = X["i_id"].astype(np.int32)
    X["timestamps"] = pd.to_datetime(X['timestamps'])
    return X


    # X=X[["u_id", "i_id", "rating", "t"]].values
X = _preprocess_data(train)
X_test = _preprocess_data(test)
def _tu_and_edulive_time(X,X_test,alpha,beta):

    print(X.groupby(["u_id"])["timestamps"])
    X['tu'] = X.groupby(["u_id"])["timestamps"].transform('mean')
    X_test["tu"] = X.groupby(["u_id"])["timestamps"].transform('mean')
    print(X_test.isnull().sum().sum())
    X['edulive_time']=alpha*(np.log(1+np.abs((X["timestamps"]-X["tu"]).astype('timedelta64[D]'))**beta))
    X_test['edulive_time']=alpha*(np.log(1+np.abs((X_test["timestamps"]-X["tu"]).astype('timedelta64[D]'))**beta))
    X_test.fillna(0, inplace=True)
    return X[["u_id","i_id","rating","edulive_time"]].values, X_test[["u_id","i_id","rating","edulive_time"]].values
_tu_and_edulive_time(X,X_test,0.0001, 0.0001)