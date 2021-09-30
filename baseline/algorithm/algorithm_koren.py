import numpy as np
from numba import jit
import pandas as pd
import time


__all__ = [
    "_compute_val_metrics",
    "_initialization",
    "_run_epoch",
]


def _initialization(n_users, n_items, n_features, rand_type, deviation):

    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    if rand_type == 1:
        pu = np.random.uniform(
            low=-deviation, high=deviation, size=(n_users, n_features)
        )
        qi = np.random.uniform(
            low=-deviation, high=deviation, size=(n_items, n_features)
        )
    else:
        pu = np.random.normal(loc=0.0, scale=deviation, size=(n_users, n_features))
        qi = np.random.normal(loc=0.0, scale=deviation, size=(n_items, n_features))

    return bu, bi, pu, qi


@jit
def _run_epoch(X, bu, bi, pu, qi, global_mean_, lr, reg, mod, std_):
    """Runs an epoch, updating model weights (pu, qi, bu, bi).

    Parameters
    ----------
    X : numpy.array
        Training set.
    bu : numpy.array
        User biases vector.
    bi : numpy.array
        Item biases vector.
    pu : numpy.array
        User latent features matrix.
    qi : numpy.array
        Item latent features matrix.
    global_mean : float
        Ratings arithmetic mean.
    n_features : int
        Number of latent features.
    lr : float
        Learning rate.
    reg : float
        L2 regularization feature.

    Returns:
    --------
    bu : numpy.array
        User biases vector.
    bi : numpy.array
        Item biases vector.
    pu : numpy.array
        User latent features matrix.
    qi : numpy.array
        Item latent features matrix.
    Lost Func:
    J(0) = ( r_ui - (bu + bi + pu^T*qi) + reg*(||pu||^2 + qi||qi||^2 + bu^2 + bi^2)
    update params:

    """
    n_shape = X.shape[0]
    for i in range(n_shape):
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
        # Predict current rating
        pred = global_mean_ + bu[user] + bi[item] + pu[user].T @ qi[item]
        if mod == "sqrt":
            err = np.sqrt(rating) - pred
        elif mod == "norm":
            err = (rating - global_mean_) / std_ - pred
        else:
            err = rating - pred
        bu[user] += lr * (err - reg * bu[user])
        bi[item] += lr * (err - reg * bi[item])
        pu[user] += lr * (err * qi[item] - reg * pu[user])
        qi[item] += lr * (err * pu[user] - reg * qi[item])

    return bu, bi, pu, qi


@jit
def _compute_val_metrics(X, bu, bi, pu, qi, global_mean_, mod, std_):
    """Computes validation metrics (loss, rmse, and mae).

    Parameters
    ----------
    X_val : numpy.array
        Validation set.
    bu : numpy.array
        User biases vector.
    bi : numpy.array
        Item biases vector.
    pu : numpy.array
        User latent features matrix.
    qi : numpy.array
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
    bu[-1] = 0
    bi[-1] = 0
    pu[-1] = 0
    qi[-1] = 0

    for i in range(n_shape):
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
        pred = global_mean_ + bu[user] + bi[item] + pu[user].T @ qi[item]
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


@jit
def _compute_val_metrics_average(
    X, bu, bi, pu, qi, global_mean_, pu_1, qi_1, mod, std_
):
    pu[-1] = pu_1
    qi[-1] = qi_1
    bu[-1] = 0
    bi[-1] = 0
    residuals = []
    n_shape = X.shape[0]
    for i in range(n_shape):
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
        pred = global_mean_ + bu[user] + bi[item] + pu[user].T @ qi[item]

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


@jit
def _compute_val_metrics_bubi_average(
    X, bu, bi, pu, qi, global_mean_, pu_1, qi_1, bu_1, bi_1, mod, std_
):
    bu[-1] = bu_1
    bi[-1] = bi_1
    pu[-1] = pu_1
    qi[-1] = qi_1
    residuals = []

    n_shape = X.shape[0]
    for i in range(n_shape):
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
        pred = global_mean_ + bu[user] + bi[item] + pu[user].T @ qi[item]
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


# def _compute_val_metrics_bypass(X, bu, bi, pu, qi, global_mean):
#     X = np.delete(X, np.where(X[:, 0] == -1), axis=0)
#     X = np.delete(X, np.where(X[:, 1] == -1), axis=0)
#     n_shape = X.shape[0]
#     residuals = []
#     start = time.process_time_ns()
#     for i in range(n_shape):
#         user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
#         pred = global_mean + bu[user] + bi[item] + pu[user].T @ qi[item]
#         residuals.append(rating - pred)
#     end = time.process_time_ns()
#     process_time = (end - start) / len(residuals)
#     residuals = np.array(residuals)
#     loss = np.square(residuals).mean()
#     rmse = np.sqrt(loss)
#     mae = np.absolute(residuals).mean()
#     return loss, rmse, mae, process_time
