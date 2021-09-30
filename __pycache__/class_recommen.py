from flask_restful import Resource
from flask import request , jsonify
import json
import numpy as np
import pickle
from predict import time_days, time_edulive, predict, update_param



class Update_Params_And_Recommend(Resource):
    def get(self):
        receive_request_json = request.json
        print(receive_request_json)
        u_id = receive_request_json["u_id"]
        i_id = receive_request_json["i_id"]
        rating = receive_request_json["rating"]
        timestamps = receive_request_json["timestamps"]
        time_exam = receive_request_json["time_exam"]
        action_exam= receive_request_json["action_exam"]


        params = open(r"C:\Users\nvtru\Desktop\auto-recommender\resuld\movielens_edulive_params",'rb')
        hyperparams = open(r"C:\Users\nvtru\Desktop\auto-recommender\resuld\movielens_edulive_hyperparams",'rb')
        user_item_map = open(r"C:\Users\nvtru\Desktop\auto-recommender\resuld\movielens_edulive_params_user_item_map",'rb')

        params = pickle.load(params)
        user_item_map = pickle.load(user_item_map)
        hyperparams = pickle.load(hyperparams)

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
        action_exam_mean=params["action_exam_mean_"]
        tu = params["tu"]
        ##################################################
        #hyper-params
        lambda1 =hyperparams["lambda1"]
        lambda2 = hyperparams['lambda2']
        alpha=hyperparams['alpha']
        beta = hyperparams['beta']
        reg = hyperparams["reg"]
        pen = hyperparams["pen"]
        mod = hyperparams["mod"]
        rand_type = hyperparams["rand_type"]
        lr = hyperparams["lr"]
        n_features = hyperparams["n_features"]
        #################################################
        user_map = user_item_map['user_map']
        item_map = user_item_map['item_map']

        if u_id not in user_map:
            user_map[u_id] = len(user_map)
            print(user_map)
            if rand_type == 1:
                pu = np.append(pu, np.random.uniform(low=-0.1, high=0.1, size=(1, n_features)), axis=0)
            else:
                pu = np.append(pu, np.random.normal(loc=0.0, scale=0.1, size=(1, n_features)), axis=0)
        if i_id not in item_map:
            item_map[i_id] = len(item_map)
            print(user_map)
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
              action_exam_mean=action_exam_mean,
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
        return predict(user=user,
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
              recommend="higher",
              item_map=item_map)
        