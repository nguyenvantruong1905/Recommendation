from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space.space import Real, Integer, Categorical
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss
import pickle
import pandas as pd
import numpy as np
path_file_train = (r"C:\Users\nvtru\Desktop\AI-Recommender\movieslen_data\train.txt")
path_file_test = (r"C:\Users\nvtru\Desktop\AI-Recommender\movieslen_data\test.txt")
path_file_val  = (r"C:\Users\nvtru\Desktop\AI-Recommender\movieslen_data\val.txt")
train = pd.read_csv(path_file_train, sep=",",
                names=["u_id", "i_id", "rating","timestamps"])
test = pd.read_csv(path_file_test, sep=",",
                names=["u_id", "i_id", "rating","timestamps"])
val = pd.read_csv(path_file_val, sep=",",
                names=["u_id", "i_id", "rating","timestamps"])

# print(map_user_item)
file_params = open(r"C:\Users\nvtru\Desktop\auto-recommender\moveliens_andrew_nomod_params","rb")
params = pickle.load(file_params)
pu = params[0]
qi = params[1]
def _preprocess_train(X):
    user_ids = X["u_id"].unique().tolist()
    item_ids = X["i_id"].unique().tolist()
    n_users = len(user_ids)
    n_items = len(item_ids)
    user_idx = range(n_users)
    item_idx = range(n_items)
    user_mapping_ = dict(zip(user_ids, user_idx))
    item_mapping_ = dict(zip(item_ids, item_idx))
    return user_mapping_, item_mapping_


def _preprocess_val(X, user_mapping_, item_mapping_):
    X["u_id"] = X["u_id"].map(user_mapping_)
    X["i_id"] = X["i_id"].map(item_mapping_)
    X.fillna(-1, inplace=True)
    X["u_id"] = X["u_id"].astype(np.int32)
    X["i_id"] = X["i_id"].astype(np.int32)
    X = X[X["rating"]!=3]
    return X[["u_id", "i_id",]].values

user_mapping_, item_mapping_ =  _preprocess_train(train)
train = _preprocess_val(train, user_mapping_, item_mapping_)

test = _preprocess_val(test, user_mapping_, item_mapping_)

def pred_X(X):
    predict_X = []
    n_row = X.shape[0]
    for i in range(n_row):
        user, item = int(X[i, 0]), int(X[i, 1])
        pred = pu[user].T @ qi[item]
        predict_X.append(pred)
    return predict_X
predict_train = pred_X(train)
predict_test = pred_X(test)
Y_train = [1 if i>3 else 0 for i in predict_train]
Y_test = [1 if i>3 else 0 for i in predict_test]
X_train = np.array(predict_train).reshape(-1, 1)
X_test = np.array(predict_test).reshape(-1, 1)


space_logisticregression = [
                Real(1, 1e7, "uniform", name="C"),
            ]
@use_named_args(space_logisticregression)
def log_loss_movielens(C):
    print("C = ", C, end=" - ")
    logreg = LogisticRegression(class_weight="balanced",C=C)
    logreg.fit(X_train,Y_train)
    y_pred_pro  = logreg.predict_proba(X_test)
    logloss = log_loss(Y_test,y_pred_pro)
    print("log_loss = ",logloss)
    return logloss
@use_named_args(space_logisticregression)
def  roc_auc_score_movielens(C):
    print("C = ", C, end=" - ")
    logreg = LogisticRegression(class_weight="balanced",C=C)
    logreg.fit(X_train,Y_train)
    y_pred_pro  = logreg.predict_proba(X_test)
    print(y_pred_pro)
    print("roc_auc_score = ", roc_auc_score(Y_test,y_pred_pro[:,1]))
    return 1 / roc_auc_score(Y_test,y_pred_pro[:,1])
res = gp_minimize(func=roc_auc_score_movielens,dimensions=space_logisticregression,n_calls=30)
print(1/res.fun, res.x)