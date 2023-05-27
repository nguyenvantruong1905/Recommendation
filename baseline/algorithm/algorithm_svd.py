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

    return pu, qi


@jit
def _run_epoch(X, pu, qi, lr, reg, mod, global_mean_, std_):
    """Runs an epoch, updating model weights (pu, qi, bu, bi).

    Parameters
    ----------
    X : numpy.array
        Training set.
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
    pu : numpy.array
        User latent features matrix.
    qi : numpy.array
        Item latent features matrix.

    -----
    lost func:
    J(0) = sigma((r_ui - pu^T*qi) + reg*(||pu||^2 + ||qi||^2))
    -----
    update params:
    pu := pu - lr* ((r_ui - pu^T*qi) + reg*(pu))
    qi := qi - lr* ((r_ui - pu^T*qi) + reg*(qi))

    """
    n_shape = X.shape[0]
    for i in range(n_shape):
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
        pred = pu[user].T @ qi[item]
        if mod == "sqrt":
            err = np.sqrt(rating) - pred
        elif mod == "norm":
            err = (rating - global_mean_) / std_ - pred
        else:
            err = rating - pred
        pu[user] += lr * (err * qi[item] - reg * pu[user])
        qi[item] += lr * (err * pu[user] - reg * qi[item])
    return pu, qi


@jit
def _compute_val_metrics(X, pu, qi, pu_1, qi_1, mod, global_mean_, std_):
    """Computes validation metrics (loss, rmse, and mae).

    Parameters
    ----------
    X_val : numpy.array
        Validation set.
    pu : numpy.array
        User latent features matrix.
    qi : numpy.array
        Item latent features matrix.
    global_mean : float
        Ratings arithmetic mean.
    n_features : int
        Number of latent features.

    Returns
    -------
    loss, rmse, mae : tuple of floats
        Validation loss, rmse and mae.
    """
    pu[-1] = pu_1
    qi[-1] = qi_1
    residuals = []

    n_shape = X.shape[0]
    for i in range(n_shape):
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
        pred = pu[user].T @ qi[item]
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


# def _compute_val_metrics_bypass(X, pu, qi):
#     user_new = 0
#     item_new = 0
#     residuals = []
#     start = time.process_time_ns()
#     n_shape = X.shape[0]
#     for i in range(n_shape):
#         user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
#         # if user == -1:
#         if user == -1:
#             user_new = user_new + 1
#             continue
#         if item == -1:
#             item_new = item_new + 1
#             continue

#         pred = pu[user].T @ qi[item]
#         residuals.append(rating - pred)
#     end = time.process_time_ns()
#     process_time = (end - start) / len(residuals)
#     residuals = np.array(residuals)
#     loss = np.square(residuals).mean()
#     rmse = np.sqrt(loss)
#     mae = np.absolute(residuals).mean()
#     print("new_user: ", user_new, " - ", "new_item: ", item_new)
#     return loss, rmse, mae, process_time
