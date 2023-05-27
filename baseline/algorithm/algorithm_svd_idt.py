import numpy as np
import time
import pandas as pd
from numba import jit


__all__ = [
    "_compute_val_metrics",
    "_initialization",
    "_run_epoch",
]


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
def _run_epoch(X, pu, qi, pu_a, qi_a, global_mean_, pen, lr, reg, mod, std_):

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
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
        pred = (
            global_mean_
            + pu[user].T @ (qi_a)
            + qi[item].T @ (pu_a)
            + qi[item].T @ pu[user]
        )
        if mod == "sqrt":
            err = np.sqrt(rating) - pred
        elif mod == "norm":
            err = (rating - global_mean_) / std_ - pred
        else:
            err = rating - pred

        pu_a += lr * (
            err * (qi[item])
            - reg * pu_a
            - pen * (-qi_a) * (global_mean_ - pu_a.T @ qi_a)
        )
        qi_a += lr * (
            err * (pu[user].T)
            - reg * qi_a
            - pen * (-pu_a) * (global_mean_ - pu_a.T @ qi_a)
        )

        pu[user] += lr * (err * (qi_a + qi[item]) - reg * pu[user])
        qi[item] += lr * (err * (pu_a + pu[user]) - reg * qi[item])
    return pu, qi, pu_a, qi_a


@jit
def _compute_val_metrics(X, pu, qi, pu_a, qi_a, global_mean_, mod, std_):
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
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
        pred = (
            global_mean_
            + pu[user].T @ (qi_a)
            + qi[item].T @ (pu_a)
            + qi[item].T @ pu[user]
        )
        if mod == "sqrt":
            residuals.append(rating - pred * pred)
        elif mod == "norm":
            residuals.append(rating - (pred * std_ + global_mean_))
        else:
            residuals.append(rating - pred)
    residuals = np.array(residuals)
    loss = np.square(residuals).mean()
    rmse = np.sqrt(loss)
    mae = np.absolute(residuals).mean()
    return loss, rmse, mae


# @jit
def _compute_val_metrics_average(
    X, pu, qi, pu_a, qi_a, global_mean_, pu_1, qi_1, mod, std_
):
    residuals = []
    pu[-1] = pu_1
    qi[-1] = qi_1
    n_shape = X.shape[0]
    for i in range(n_shape):
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
        pred = (
            global_mean_
            + pu[user].T @ (qi_a)
            + qi[item].T @ (pu_a)
            + qi[item].T @ pu[user]
        )
        if mod == "sqrt":
            residuals.append(rating - pred * pred)
        elif mod == "norm":
            residuals.append(rating - (pred * std_ + global_mean_))
        else:
            residuals.append(rating - pred)

    residuals = np.array(residuals)
    loss = np.square(residuals).mean()
    rmse = np.sqrt(loss)
    mae = np.absolute(residuals).mean()
    return loss, rmse, mae


# def _compute_val_metrics_bypass(X, pu, qi, pu_a, qi_a, global_mean_):
#     start = time.process_time_ns()
#     residuals = []

#     X = np.delete(X, np.where(X[:, 0] == -1), axis=0)
#     X = np.delete(X, np.where(X[:, 1] == -1), axis=0)
#     n_shape = X.shape[0]
#     for i in range(n_shape):
#         user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
#         pred = (
#             global_mean_
#             + pu[user].T @ (qi_a)
#             + qi[item].T @ (pu_a)
#             + qi[item].T @ pu[user]
#         )
#         residuals.append(rating - pred)
#     end = time.process_time_ns()
#     process_time = (end - start) / len(residuals)
#     residuals = np.array(residuals)
#     loss = np.square(residuals).mean()
#     rmse = np.sqrt(loss)
#     mae = np.absolute(residuals).mean()
#     return loss, rmse, mae, process_time
