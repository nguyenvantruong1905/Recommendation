from algorithm import Andrew
from algorithm import Edulive
from algorithm import Koren


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

    path_file_train = (r"C:\Users\nvtru\Desktop\auto-recommender\ml-1m\train.txt")
    path_file_test = (r"C:\Users\nvtru\Desktop\auto-recommender\ml-1m\test.txt")
    train = pd.read_csv(path_file_train, sep=",",
                    names=["u_id", "i_id", "rating","timestamps"])
    test = pd.read_csv(path_file_test, sep=",",
                    names=["u_id", "i_id", "rating","timestamps"])

    #########################################################
    # optimization_Andrew
    space_SVD_Andrew = [
        Real(1e-5, 1e0, "uniform", name="reg"),
        Integer(1, 100, "uniform", name="n_features"),
        Integer(0, 1, "uniform", name="rand_type"),
    ]

    @use_named_args(space_SVD_Andrew)
    def f_SVD_Andrew(reg, n_features, rand_type):
        print(reg, n_features, rand_type)
        try:
            res_ = load(path + filename_pkl)
            min_rmse = res_.fun
        except:
            min_rmse = 100000.0
        print(min_rmse)
        estimator = SVD_Andrew(
            lr=lr,
            reg=reg,
            n_features=n_features,
            n_epochs=num_epochs,
            rand_type=rand_type,
            deviation=0.1,
            algorithm="andrew",
            mod=mod,
            storefile=True,
            filename=path + filename_params,
            min_rmse=min_rmse,
        )
        a = estimator.fit(train, test)

        return a

    #########################################################
    # optimization_Edulive
    space_SVD_Edulive = [
        Real(1e-5, 1e0, "uniform", name="reg"),
        Real(1e-4, 1e2, "uniform", name="pen"),
        Integer(3, 3.1, "uniform", name="n_features"),
        Integer(0, 1, "uniform", name="rand_type"),
    ]

    @use_named_args(space_SVD_Edulive)
    def f_SVD_Edulive(reg, pen, n_features, rand_type):
        print(reg, pen, n_features, rand_type)
        try:
            res_ = load(path + filename_pkl)
            min_rmse = res_.fun
        except:
            min_rmse = 100000.0
        print(min_rmse)
        estimator = SVD_Edulive(
            lr=lr,
            reg=reg,
            pen=pen,
            n_features=n_features,
            n_epochs=num_epochs,
            rand_type=rand_type,
            deviation=0.1,
            algorithm="edulive",
            mod=mod,
            storefile=True,
            filename=path + filename_params,
            min_rmse=min_rmse,
        )
        a = estimator.fit(train, test)

        return a

    ##############################################################
    # optimization_Koren
    space_SVD_Koren = [
        Real(1e-5, 1e0, "uniform", name="reg"),
        Integer(1, 100, "uniform", name="n_features"),
        Integer(0, 1, "uniform", name="rand_type"),
    ]

    @use_named_args(space_SVD_Koren)
    def f_SVD_Koren(reg, n_features, rand_type):
        print(reg, n_features, rand_type)
        # print(train,test)
        try:
            res_ = load(path + filename_pkl)
            min_rmse = res_.fun
        except:
            min_rmse = 100000.0
        print(min_rmse)
        estimator = SVD_Koren(
            lr=lr,
            reg=reg,
            n_features=n_features,
            n_epochs=2,
            rand_type=rand_type,
            deviation=0.1,
            algorithm="koren",
            mod=mod,
            storefile=True,
            filename=path + filename_params,
            min_rmse=min_rmse,
        )
        a = estimator.fit(train, test)

        return a

    ##############################################################
    num_epochs = 3000
    mod = "nomod"  # "norm", "sqrt", "nomod"
    lr = 1e-3
    alg = "andrew"
    dataset = "moveliens"
    prefix = dataset + "_" + alg
    if alg == "koren":
        func = f_SVD_Koren
        dimensions = space_SVD_Koren
    elif alg == "andrew":
        func = f_SVD_Andrew
        dimensions = space_SVD_Andrew
    elif alg == "edulive":
        func = f_SVD_Edulive
        dimensions = space_SVD_Edulive

    ##############################################################

    filename_txt = prefix + "_" + mod + ".txt"
    filename_pkl = prefix + "_" + mod + ".pkl"
    filename_params = prefix + "_" + mod + "_params"
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
        dimensions=dimensions,
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
