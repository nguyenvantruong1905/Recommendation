import pickle
import pandas as pb
import numpy as np
from datetime import datetime

from datetime import datetime


max_rating = 5
params = open(r"/home/truong/auto-recommender/movielens_edulive_nomod_params",'rb')
hyperparams = open(r"/home/truong/auto-recommender/resuld/movielens_edulive_nomod_hyperparams",'rb')
user_item_map = open(r"/home/truong/auto-recommender/resuld/movielens_edulive_nomod_params_user_item_map",'rb')

params = pickle.load(params)
user_item_map = pickle.load(user_item_map)
hyperparams = pickle.load(hyperparams)
# print(params)
# print(hyperparams)
# print(user_item_map)
# user_map = user_item_map['user_map']
# item_map = user_item_map['item_map']
###################################################
#params:
pu = params['pu']
qi = params["qi"]
pu_a = params["pu_a"]
qi_a = params["qi_a"]
tu = params["tu"]
global_mean = params["global_mean_"]
std = params["std_"]
time_exam_mean = params["time_exam_mean_"]
action_exam_mean_=params["action_exam_mean_"]
tu = params["tu"]
##################################################
#hyper-params
lambda1 =hyperparams["lambda1"]
lambda2 = hyperparams['lambda2']
alpha=hyperparams['alpha']
beta = hyperparams['beta']
reg = hyperparams["reg"]
pen = hyperparams["pen"]
rand_type = hyperparams["rand_type"]
lr = hyperparams["lr"]
mod=hyperparams["mod"]
n_features = hyperparams["n_features"]
#################################################
user_map = user_item_map['user_map']
item_map = user_item_map['item_map']



def update_param(pu, qi, pu_a,qi_a, user, item, rating,time_exam_mean,action_exam_mean, timestamps,time_exam,action_exam,tu,global_mean_,lambda1,lambda2, alpha,beta, mod, lr , reg, pen, std_ ):
    # print(user_map)
    pred = (
            global_mean_
            + (pu[user] + time_edulive(timestamps,tu[user],alpha,beta)).T @ (qi_a)
            + qi[item].T @ (pu_a)
            + qi[item].T @ (pu[user] + time_edulive(timestamps,tu[user],alpha,beta))
            - lambda1*time_exam
            - lambda2*action_exam
    )
    if mod == "sqrt":
        err = np.sqrt(rating) - pred
    elif mod == "norm":
        err = (rating - global_mean_) / std_ - pred
    elif mod == "log":
        err = np.log(1+rating) - pred
    else:
        err = rating - pred
    pu_a += lr * (
        err * (qi[item])
        - reg * pu_a
        - pen * (-qi_a) * (global_mean_ - pu_a.T @ qi_a)
    )
    qi_a += lr * (
        err * ((pu[user] + time_edulive(timestamps,tu[user],alpha,beta)))
        - reg * qi_a
        - pen * (-pu_a) * (global_mean_ - pu_a.T @ qi_a)
    )

    pu[user] += lr * (err * (qi_a + qi[item]) - reg * pu[user])
    qi[item] += lr * (err * (pu_a + (pu[user] +time_edulive(timestamps,tu[user],alpha,beta))) - reg * qi[item])
    data = {
        "pu" : pu,
        "qi" : qi,
        "pu_a":pu_a,
        "qi_a":qi_a,
        "global_mean_": global_mean_,
        "std_":std_,
        "time_exam_mean_":time_exam_mean,
        "action_exam_mean_":action_exam_mean,
        "tu": tu
    }
    dbfile = open(r"/home/truong/auto-recommender/movielens_edulive_nomod_params",'wb')
    pickle.dump(data, dbfile)

    return pu , qi , pu_a, qi_a, global_mean_,std_, time_exam_mean,action_exam_mean,tu
    
def time_edulive(timestamps, tu,alpha,beta):
    ts = abs(timestamps - tu)
    # print(ts.days)
    return alpha*np.log(1+(ts.days)**beta)

def time_days(tu):
    dt = datetime.utcnow()
    dt64 = np.datetime64(dt)
    ts = abs(dt64 - tu)
    return(ts.days)#astype('timedelta64[D]')/ np.timedelta64(1, 'D'))

def predict(user, pu, qi, pu_a, qi_a,time_exam_mean,tu, action_exam_mean, global_mean, lambda1, lambda2, alpha,mod, beta , std, recommend = None):
    # user = user_map[1604]
    arr =[]
    max_rating = 5
    for item in range(qi.shape[0]):
        pred = (
        global_mean
        + (pu[user] + alpha*(np.log(1+(time_days(tu[user])**beta)))) .T @ (qi_a)
        + qi[item].T @ (pu_a)
        + qi[item].T @ (pu[user] + alpha*(np.log(1+(time_days(tu[user])**beta))))
        - lambda1*time_exam_mean
        - lambda2*action_exam_mean
    )
        if mod == "sqrt":
            arr.append(pred * pred)
        elif mod == "norm":
            arr.append((pred * std + global_mean))
        elif mod == "log":
            arr.append((np.exp(pred) - 1))
        else:
            arr.append(pred)
    arr_higher = np.array(arr)
    arr_improve = max_rating-arr_higher
    arr_map = np.argsort(arr_higher)
    recommend_item1 =  [k for k,v in item_map.items() if v==arr_map[0]]
    recommend_item2 =  [k for k,v in item_map.items() if v==arr_map[1]]
    arr_map = np.argsort(arr_improve)
    recommend_item3 =  [k for k,v in item_map.items() if v==arr_map[0]]
    recommend_item4 =  [k for k,v in item_map.items() if v==arr_map[1]]

    return {
        "higher":[recommend_item1,recommend_item2],
        "improve":[recommend_item3,recommend_item4],
    }

u_id = 1604
i_id = 2533
rating = 4 
timestamps = datetime.now()
time_exam = 32
action_exam= 46
if u_id not in user_map:
    user_map[u_id] = len(user_map)
    if rand_type == 1:
        pu = np.append(pu, np.random.uniform(low=-0.1, high=0.1, size=(1, n_features)), axis=0)
    else:
        pu = np.append(pu, np.random.normal(loc=0.0, scale=0.1, size=(1, n_features)), axis=0)
if i_id not in item_map:
    item_map[i_id] = len(item_map)
    if rand_type == 1:
        qi = np.append(qi, np.random.uniform(low=-0.1, high=0.1, size=(1, n_features)), axis=0)
    else:
        qi = np.append(qi, np.random.normal(loc=0.0, scale=0.1, size=(1, n_features)), axis=0)

user = user_map[u_id]
item = item_map[i_id]
pu , qi , pu_a, qi_a, global_mean,std, time_exam_mean,action_exam_mean,tu =update_param( pu = pu,user=user, item=item,
              qi = qi,
              qi_a=qi_a,
              pu_a=pu_a,
              rating=rating,
              timestamps=timestamps,
              time_exam=time_exam,
              time_exam_mean=time_exam_mean,
              action_exam_mean=action_exam_mean_,
              action_exam=action_exam,
              tu = tu,
              global_mean_= global_mean,
              lambda1=lambda1,
              lambda2= lambda2,
              alpha= alpha,
              beta=beta,
              mod=mod,
              lr=lr,
              reg= reg,
              pen=pen,
              std_ = std,
              )
print(predict(user=user,
              pu = pu,
              qi = qi,
              qi_a=qi_a,
              pu_a=pu_a,
              time_exam_mean=time_exam_mean,
              action_exam_mean=action_exam_mean,
              tu = tu,
              global_mean= global_mean,
              lambda1=lambda1,
              lambda2= lambda2,
              alpha= alpha,
              beta=beta,
              mod=mod,
              std= std,
              recommend="higher"

))
