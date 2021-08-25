import numpy as np
import time
import pandas as pd
from numba import jit


__all__ = [
    "_compute_val_metrics",
    "_initialization",
    "_run_epoch",
]

def _tu_and_edulive_time(X,X_test,alpha,beta):
    # X["timestamps"] = pd.to_datetime(X["timestamps"])
    # print(X)
    # a = X.groupby('u_id').mean()["timestamps"]
    # print(a)
    # X = pd.merge(X,pd.DataFrame(X.groupby('u_id').mean()["timestamps"]), on= "u_id", how="left")
    # print((X["timestamps_x"]-X["timestamps_y"]).astype('timedelta64[D]'))
    # X_test = pd.merge(X_test,pd.DataFrame(X.groupby('u_id').mean()["timestamps"]), on= "u_id", how="left")
    # X['edulive_time']=alpha*(np.log(1+np.abs((X["timestamps_x"]-X["timestamps_y"]).astype('timedelta64[D]'))**beta))
    # X_test['edulive_time']=alpha*(np.log(1+np.abs((X_test["timestamps_x"]-X_test["timestamps_y"]).astype('timedelta64[D]'))**beta))
    # X_test.fillna(0,inplace=True)
    X["timestamps"] = pd.to_datetime(X["timestamps"]).values.astype(np.int64)
    df = pd.to_datetime(X.groupby('u_id').mean()["timestamps"])
    X = pd.merge(X,df, on= ["u_id"],how="left")
    X.columns=["u_id","i_id","rating","t","time_exam", "action_exam","tu"]
    X["t"] = pd.to_datetime(X['t'])
    X['edulive_time']=alpha*(np.log(1+np.abs((X["t"]-X["tu"]).astype('timedelta64[D]'))**beta))
    X_test["timestamps"] = pd.to_datetime(X_test["timestamps"]).values.astype(np.int64)
    X_test = pd.merge(X_test,df, on= ["u_id"],how="left")
    X_test.columns=["u_id","i_id","rating","t","time_exam", "action_exam","tu"]
    X_test["t"] = pd.to_datetime(X['t'])
    X_test['edulive_time']=alpha*(np.log(1+np.abs((X["t"]-X["tu"]).astype('timedelta64[D]'))**beta))
    X.fillna(0, inplace=True)

    return X[["u_id","i_id","rating","time_exam", "action_exam","edulive_time"]].values,X[["u_id","i_id","rating","time_exam", "action_exam","edulive_time"]].values,X["tu"].to_list()

def _initialization(n_users, n_items, n_features, rand_type, deviation):

    if rand_type == 1:
        pu = np.random.uniform(
            low=-deviation, high=deviation, size=(n_users, n_features)
        )
        qi = np.random.uniform(
            low=-deviation, high=deviation, size=(n_items, n_features)
        )
        pu_a = np.random.uniform(low=-deviation, high=deviation, size=n_features)
        qi_a = np.random.uniform(low=-deviation, high=deviation, size=n_features)
    else:
        pu = np.random.normal(loc=0.0, scale=deviation, size=(n_users, n_features))
        qi = np.random.normal(loc=0.0, scale=deviation, size=(n_items, n_features))
        pu_a = np.random.normal(loc=0.0, scale=deviation, size=n_features)
        qi_a = np.random.normal(loc=0.0, scale=deviation, size=n_features)

    return pu, qi, pu_a, qi_a


@jit
def _run_epoch(X, pu, qi, pu_a, qi_a, global_mean_, pen, lr, reg, mod, std_,  lambda1 , lambda2):

    """Runs an epoch, updating model weights (pu_a, qi_a, pu, qi).

    Parameters
    ----------
    X : numpy.array
        Training set.
    pu : numpy.array
        User qiases vector.
    qi : numpy.array
        Item qiases vector.
    pu_a : numpy.array
        User latent features matrix.
    qi_a : numpy.array
        Item latent features matrix.
    global_mean_ : float
        Ratings arithmetic mean.
    n_features : int
        Number of latent features.
    lr : float
        Learning rate.
    reg : float
        L2 regularization feature.

    Returns:
    --------
    pu : numpy.array
        User qiases vector.
    qi : numpy.array
        Item qiases vector.
    pu_a : numpy.array
        User latent features matrix.
    qi_a : numpy.array
        Item latent features matrix.

    j(0) = sima( r_{ui} - ( M  +  pu^T*qi_a  +  qi^T*pu_a   +   qi^T*pu ) ) + reg * ( ||qi||^2 + ||pu||^2 + ||pu_a||^2 + ||qi||^2 ) + pen * ( M - pu^T*qi )^2)
    update params :
    pu_a :=  pu_a - lr * ( (r_{ui} - ( M  +  pu^T*qi_a  +  qi^T*pu_a   +   qi^T*pu))*(-qi) ) + reg*(pu_a)  +  pen*(M - pu_a^T*qi_a)*(-qi_a)))
    qi_a :=  qi_a - lr * ( (r_{ui} - ( M  +  pu^T*qi_a  +  qi^T*pu_a   +   qi^T*pu))*(-pu^T) + reg*(qi_a)  +  pen*(M - pu_a^T*qi_a)*(-pu_a)))
    pu   :=  pu   - lr * ( (r_{ui} - ( M  +  pu^T*qi_a  +  qi^T*pu_a   +   qi^T*pu))*(-qi^T - qi) + reg*(pu))
    qi   :=  qi   - lr * ( (r_{ui} - ( M  +  pu^T*qi_a  +  qi^T*pu_a   +   qi^T*pu))*(-pu^T - pu_a^T ) + reg*(qi))
    """
    n_shape = X.shape[0]
    for i in range(n_shape):
        user, item, rating, time_exam, action,edulive_time = int(X[i, 0]), int(X[i, 1]), X[i, 2], X[i,3], X[i,4], X[i,5]

        pred = (
            global_mean_
            + (pu[user] + edulive_time).T @ (qi_a)
            + qi[item].T @ (pu_a)
            + qi[item].T @ (pu[user] + edulive_time)
            - lambda1*time_exam
            - lambda2*action
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
            err * ((pu[user] + edulive_time))
            - reg * qi_a
            - pen * (-pu_a) * (global_mean_ - pu_a.T @ qi_a)
        )

        pu[user] += lr * (err * (qi_a + qi[item]) - reg * pu[user])
        qi[item] += lr * (err * (pu_a + (pu[user] +edulive_time)) - reg * qi[item])
    return pu, qi, pu_a, qi_a


@jit
def _compute_val_metrics(X, pu, qi, pu_a, qi_a, global_mean_, mod, std_ ,  lambda1 , lambda2):
    """Compu_ates validation metrics (loss, rmse, and mae).

    Parameters
    ----------
    X_val : numpy.array
        Validation set.
    pu : numpy.array
        User qiases vector.
    qi : numpy.array
        Item qiases vector.
    pu_a : numpy.array
        User latent features matrix.
    qi_a : numpy.array
        Item latent features matrix.
    global_mean_ : float
        Ratings arithmetic mean.
    n_features : int
        Number of latent features.

    Returns
    -------
    loss, rmse, mae : tuple of floats
        Validation loss, rmse and mae.
    """
    residuals = []
    n_shape = X.shape[0]
    pu[-1] = 0
    qi[-1] = 0
    for i in range(n_shape):
        user, item, rating, time_exam, action,edulive_time = int(X[i, 0]), int(X[i, 1]), X[i, 2], X[i,3], X[i,4], X[i,5]

        pred = (
            global_mean_
            + (pu[user] + edulive_time).T @ (qi_a)
            + qi[item].T @ (pu_a)
            + qi[item].T @ (pu[user] + edulive_time)
            - lambda1*time_exam
            - lambda2*action
        )
        if mod == "sqrt":
            residuals.append(rating - pred * pred)
        elif mod == "norm":
            residuals.append(rating - (pred * std_ + global_mean_))
        elif mod == "log":
            residuals.append(rating - (np.exp(pred) - 1))
        else:
            residuals.append(rating - pred)
    residuals = np.array(residuals)
    loss = np.square(residuals).mean()
    rmse = np.sqrt(loss)
    mae = np.absolute(residuals).mean()
    return loss, rmse, mae
