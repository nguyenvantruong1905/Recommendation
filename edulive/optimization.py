
import pickle
from SVD import SVD_Edulive
import sys
import os
import glob

import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space.space import Real, Integer, Categorical
from skopt.callbacks import CheckpointSaver
from skopt import callbacks, load

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

    # path_file_train = (r"/home/truonv/Desktop/data/movieslen_data/train.txt")
    # path_file_test = (r"/home/truonv/Desktop/data/movieslen_data/test1.txt")
    # train = pd.read_csv(path_file_train, sep=",",
    #                 names=["u_id", "i_id", "rating", "timestamps"])
    # test = pd.read_csv(path_file_test, sep=",",
    #                 names=["u_id", "i_id", "rating", "timestamps"])
    try :
        while True:
            path_file_train = (r"/home/auto-recommender/movieslen_data/train.txt")
            path_file_test = (r"/home/auto-recommender/movieslen_data/test1.txt")
            train = pd.read_csv(path_file_train, sep=",",
                            names=["u_id", "i_id", "rating", "timestamps"])
            test = pd.read_csv(path_file_test, sep=",",
                            names=["u_id", "i_id", "rating", "timestamps"])
            time_exam_train = np.random.choice(30, 80000)
            time_exam_train = np.random.choice(30, 80000)
            time_exam_test = np.random.choice(30, 20000)
            action_exam_train = np.random.choice(50, 80000)
            action_exam_test = np.random.choice(50, 20000)
            train['time_exam'] = pd.Series(time_exam_train,index=train.index)
            train['action_exam'] = pd.Series(action_exam_train,index=train.index)
            test['time_exam'] = pd.Series(time_exam_test,index=test.index)
            test['action_exam'] = pd.Series(action_exam_test,index=test.index)

            # path_file_train = (
            #     r"/home/truong/AI-Recommender/project/data/diemthidaihoc2020/train.csv"
            # )
            # path_file_test = (
            #     r"/home/truong/AI-Recommender/project/data/diemthidaihoc2020/test.csv"
            # )
            # train = pd.read_csv(path_file_train, sep=",", names=["u_id", "i_id", "rating"])
            # test = pd.read_csv(path_file_test, sep=",", names=["u_id", "i_id", "rating"])
        #########################################################

        # optimization_Edulive
            space_SVD_Edulive = [
                Real(1e-4, 1e-0, "uniform", name="reg"),
                Real(1e-2, 1e2, "uniform", name="pen"),
                Integer(3, 3.1, "uniform", name="n_features"),
                Integer(0, 1, "uniform", name="rand_type"),
                Real(1e-5, 1e-2, "uniform", name="lambda1"),
                Real(1e-5, 1e-2, "uniform", name="lambda2"),
                Real(1e-5, 1e-2, "uniform", name="alpha"),
                Real(1e-3, 1e1, "uniform", name="beta"),
            ]

            @use_named_args(space_SVD_Edulive)
            def f_SVD_Edulive(reg, pen, n_features, rand_type,lambda1, lambda2, alpha, beta):
                print(reg, pen, n_features, rand_type,lambda1, lambda2, alpha, beta)
                try:
                    res_ = load(path + filename_pkl)
                    min_rmse = res_.fun
                except:
                    min_rmse = 100000.0
                print(min_rmse)
                estimator = SVD_Edulive(
                    lr=5e-4,
                    reg=reg,
                    pen=pen,
                    n_features=n_features,
                    alpha= alpha,
                    beta= beta,
                    lambda1=lambda1,
                    lambda2= lambda2,
                    rand_type=rand_type,
                    deviation=0.1,
                    std_= 0.0,
                    n_epochs=num_epochs,
                    mod = mod,
                    storefile=True,
                    filename=path + filename_params,
                    min_rmse=min_rmse,
                )
                a = estimator.fit(train, test)

                return a
        ##############################################################
            num_epochs = 2
            mod = "nomod"  # "norm", "sqrt", "nomod"
            lr = 1e-3
            dataset = "movielens"
            prefix = dataset + "_" + "edulive"

            ##############################################################

            filename_txt = prefix + "_" +mod +"_"+ ".txt"
            filename_pkl = prefix + "_"+mod +"_"+ ".pkl"
            filename_params = prefix + "_" +mod+"_"+ "params"
            filename_hyperparams = prefix + "_" +mod+"_"+ "hyperparams"
            path = r"/home/auto-recommender/optimizeparams/"
            path_resuld = r"/home/auto-recommender/resuld/"
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
                    n_init_points = 11 - len(x_0)
                ncalls = 11 - len(x_0)
                r_state = res.random_state
                b_estimator = res.specs["args"]["base_estimator"]
            except:
                x_0 = None
                y_0 = None
                n_init_points = 10
                ncalls = 11
                r_state = None
                b_estimator = None

            res = gp_minimize(
                func=f_SVD_Edulive,
                dimensions=space_SVD_Edulive,
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
            data_hyperparam = {
                "reg":res.x[0], 
                "pen":res.x[1], 
                'n_features':res.x[2], 
                'rand_type':res.x[3],
                'lambda1':res.x[4], 
                'lambda2':res.x[5], 
                'alpha':res.x[6], 
                'beta':res.x[7], 
                "lr": lr,
                "mod":mod   
                
            }
            print(data_hyperparam)
            f_hyperparams = open(path_resuld+filename_hyperparams,"wb")
            pickle.dump(data_hyperparam,f_hyperparams)
            pickle.dump(pickle.load(open(path+filename_params,"rb")), open(path_resuld+filename_params,"wb"))
            pickle.dump(pickle.load(open(path+filename_params+"_user_item_map","rb")), open(path_resuld+filename_params+"_user_item_map","wb"))
            sys.stdout.close()
            files = glob.glob(path+"*")
            for f in files:
                os.remove(f)
    except KeyboardInterrupt:
        sys.stdout = open(path + filename_txt, "w")
        sys.stderr = sys.stdout