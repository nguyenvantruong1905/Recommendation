
import pickle
import numpy as np
import xgboost as xgb
import pandas as pd
import sys
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space.space import Real, Integer
from skopt import callbacks, load
from skopt.callbacks import CheckpointSaver


if __name__ == "__main__":

    ##################################

    # path_file_train = r"/home/truong/AI-Recommender/project/data/netflix_data/train.txt"
    # path_file_test = r"/home/truong/AI-Recommender/project/data/netflix_data/test.txt"

    # train = pd.read_csv(
    #     path_file_train, sep=",", names=["u_id", "i_id", "rating", "timestamps"]
    # )
    # test = pd.read_csv(
    #     path_file_test, sep=",", names=["u_id", "i_id", "rating", "timestamps"]
    # )
    ##################################
    # path_file_train = (
    #     r"/home/truong/AI-Recommender/project/data/diemthidaihoc2020/train.csv"
    # )
    # path_file_test = (
    #     r"/home/truong/AI-Recommender/project/data/diemthidaihoc2020/test.csv"
    # )

    # path_file_train = (
    #     r"/home/truong/AI-Recommender/project/data/sinhvienjunyi_data/train.csv"
    # )
    # path_file_test = (
    #     r"/home/truong/AI-Recommender/project/data/sinhvienjunyi_data/test.csv"
    # )

    # path_file_train = r"/home/truong/AI-Recommender/project/data/thuoc_data/train.csv"
    # path_file_test = r"/home/truong/AI-Recommender/project/data/thuoc_data/test.csv"

    path_file_train = (r"C:\Users\nvtru\Desktop\AI-Recommender\thuoc\train.csv")
    path_file_test = (r"C:\Users\nvtru\Desktop\AI-Recommender\thuoc\test.csv")
    path_file_val = (r"C:\Users\nvtru\Desktop\AI-Recommender\thuoc\val.csv")
    train = pd.read_csv(path_file_train, sep=",",
                    names=["u_id", "i_id", "rating"])
    test = pd.read_csv(path_file_test, sep=",",
                    names=["u_id", "i_id", "rating"])
    val = pd.read_csv(path_file_val, sep=",",
                names=["u_id", "i_id", "rating"])
    file_map = open(r"C:\Users\nvtru\Desktop\AI-Recommender\map_user_item","rb")
    map_user_item = pickle.load(file_map)
    # print(map_user_item)
    file_params = open(r"C:\Users\nvtru\Desktop\AI-Recommender\thuoc_edulive_nomod_params","rb")
    params = pickle.load(file_params)

    #########################################################

    def xgb_optm(train,test,val,params,map_user_item,max_depth,eta,gamma,l2,min_child_weight,colsample_bytree,max_delta_step,scale_pos_weight):
        train=train.copy()
        test=test.copy()
        val = val.copy()
        pu = params[0]+params[2]
        qi = params[1]+params[3]
        pu_dict = dict(enumerate(pu))
        qi_dict = dict(enumerate(qi))
        pu_a = np.array(params[2])
        qi_a = np.array(params[3])

        map_user = {k:j for k, j in zip(map_user_item[0].keys(),pu_dict.values())}
        map_item =  {k:j for k, j in zip(map_user_item[1].keys(),qi_dict.values())}
        train["u_id"]=train["u_id"].map(map_user)
        train["i_id"]=train["i_id"].map(map_item)
        test["u_id"]=test["u_id"].map(map_user)
        test["i_id"]=test["i_id"].map(map_item)
        test.loc[test["u_id"].isnull(), 'u_id'] = test.loc[test["u_id"].isnull(), 'u_id'].apply(lambda x: pu_a)
        test.loc[test["i_id"].isnull(), 'i_id'] = test.loc[test["i_id"].isnull(), 'i_id'].apply(lambda x: qi_a)
        val["u_id"]=val["u_id"].map(map_user)
        val["i_id"]=val["i_id"].map(map_item)
        val.loc[val["u_id"].isnull(), 'u_id'] = val.loc[val["u_id"].isnull(), 'u_id'].apply(lambda x: pu_a)
        val.loc[val["i_id"].isnull(), 'i_id'] = val.loc[val["i_id"].isnull(), 'i_id'].apply(lambda x: qi_a)
        test.isnull().sum().sum()
        X_train = np.append(train["u_id"].tolist(),np.array(train["i_id"].tolist()),axis=1)
        X_test = np.append(np.array(test["u_id"].tolist()),np.array(test["i_id"].tolist()),axis=1)
        X_train_test = np.concatenate((X_train,X_test),axis=0)
        y_train_test = train["rating"].tolist() + test["rating"].tolist()
        X_val = np.append(np.array(val["u_id"].tolist()),np.array(val["i_id"].tolist()),axis=1)
        y_val = val["rating"].tolist()
        # X_train = train[["u_id","i_id"]].values
        # y_train = train["rating"].tolist()
        # X_test = test[["u_id","i_id"]].values
        # y_test = test["rating"].tolist()
        # print(X_train)

        Dtrain = xgb.DMatrix(X_train_test, label=y_train_test)
        Dval = xgb.DMatrix(X_val, label=y_val)
        x_parameters = {
            "max_depth":max_depth,
            'eta':eta,
            "gamma":gamma,
            "lambda":l2,
            "min_child_weight":min_child_weight,
            #"subsample": subsample, #Tỷ lệ mẫu phụ của các trường hợp đào tạo. Đặt nó thành 0,5 có nghĩa là XGBoost sẽ lấy mẫu ngẫu nhiên một nửa dữ liệu đào tạo trước khi trồng cây. và điều này sẽ ngăn chặn việc trang bị quá nhiều. Việc lấy mẫu con sẽ xảy ra một lần trong mỗi lần lặp lại tăng cường.
            "colsample_bytree": colsample_bytree,
            "max_delta_step": max_delta_step,
            "scale_pos_weight":scale_pos_weight,

  

            }

        bst = xgb.train(x_parameters, Dtrain, 300,evals=[(Dval, 'data_thuoc')],early_stopping_rounds=2 )
        # print(bst)
        preds = bst.predict(Dval)
        rmse = np.sqrt(np.square(np.array(preds)-np.array(y_val)).mean())
        print(rmse)
        return rmse


    #########################################################
    # optimization_Edulive
    space_XGBoost = [

        Real(0, 1, "uniform", name="eta"),
        Integer(1, 50, "uniform", name="max_depth"),
        Real(0, 20, "uniform", name="gamma"),
        Real(1e-3, 1e1, "uniform", name="l2"),
        Real(0, 20, "uniform", name="min_child_weight"),
        # Integer(1, 1.1, "uniform", name="subsample"),
        Real(0, 1.0, "uniform", name="colsample_bytree"),
        Real(0, 20, "uniform", name="scale_pos_weight"),
        Real(0, 10, "uniform", name="max_delta_step"),

        


        
    ]

    @use_named_args(space_XGBoost)
    def f_XGboost_Edulive(max_depth,eta,gamma,l2,min_child_weight,colsample_bytree,max_delta_step,scale_pos_weight):
        print(max_depth,eta,gamma,l2,min_child_weight,colsample_bytree,max_delta_step,scale_pos_weight)
        try:
            res_ = load(path + filename_pkl)
            min_rmse = res_.fun
        except:
            min_rmse = 100000.0
        print(min_rmse)
        rmse = xgb_optm(train=train,
                        test=test,
                        val =val,
                        params=params, 
                        map_user_item=map_user_item,
                        max_depth=max_depth, 
                        eta=eta, 
                        gamma=gamma, 
                        l2=l2, 
                        min_child_weight=min_child_weight,
                        # subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        max_delta_step=max_delta_step,
                        scale_pos_weight=scale_pos_weight

                        )


        return rmse

    ##############################################################

    ##############################################################
    num_epochs = 300
    mod = "nomod"  # "norm", "sqrt", "nomod"
    lr = 1e-4
    alg = "edulive"
    dataset = "thuoc"
    prefix = dataset + "_" + alg
    func = f_XGboost_Edulive
    ##############################################################

    filename_txt = prefix + "_xgboost" + mod + ".txt"
    filename_pkl = prefix + "_xgboost" + mod + ".pkl"
    # filename_params = prefix + "_xgboost" + mod + "_params"
    path = r"C:/Users/nvtru/Desktop/AI-Recommender/"
    # path = "/tmp/"
    sys.stdout = open(path + filename_txt, "w")
    sys.stderr = sys.stdout
    pkl = CheckpointSaver(path + filename_pkl, compress=9)

    try:
        res = load(path + filename_pkl)
        x_0 = res.x_iters
        y_0 = res.func_vals
        if len(x_0) >= 10:
            n_init_points = -len(x_0)
        else:
            n_init_points = 10 - len(x_0)
        ncalls = 60 - len(x_0)
        r_state = res.random_state
        b_estimator = res.specs["args"]["base_estimator"]
    except:
        x_0 = None
        y_0 = None
        n_init_points = 10
        ncalls = 60
        r_state = None
        b_estimator = None

    res = gp_minimize(
        func=func,
        dimensions=space_XGBoost,
        n_calls=ncalls,
        verbose=True,
        x0=x_0,
        y0=y_0,
        n_initial_points=n_init_points,
        random_state=r_state,
        base_estimator=b_estimator,
        n_jobs=1,
        callback=[pkl],
    )
    print(res.fun, res.x)
    sys.stdout.close()
