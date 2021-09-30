import numpy as np
import pandas as pd
import time
import pickle
from time import process_time


from .algorithm_edulive import (
    _compute_val_metrics,
    _initialization,
    _run_epoch,
    _compute_val_metrics_average,
)
from .utils import _timer

__all__ = ["SVD"]


class SVD_Edulive:
    def __init__(
        self,
        lr=0.005,
        reg=0.02,
        pen=10,
        n_features=100,
        n_epochs=300,
        rand_type=0,
        deviation=0.1,
        algorithm="edulive",
        filename="",
        storefile=False,
        mod="sqrt",
        min_rmse=10000.0,
    ):

        self.lr = lr
        self.pen = pen
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_features = n_features
        self.rand_type = rand_type
        self.deviation = deviation
        self.algorithm = algorithm
        self.filename = filename
        self.storefile = storefile
        self.mod = mod
        self.min_rmse = min_rmse

    @_timer(text="\nDone ")
    def fit(self, X, X_test=None):
        """Learns model weights from input data.
        Parameters
        ----------
        X : pandas.DataFrame
            Training set, must have 'u_id' for user ids, 'i_id' for item ids,
            and 'rating' column names.
        X_val : pandas.DataFrame, default=None
            Validation set with the same column structure as X.
        Returns
        -------
        self : SVD object
            The current fitted object.
        """
        X = self._preprocess_data(X)

        if X_test is not None:
            X_test = self._preprocess_data(X_test, train=False, verbose=False)
            self._init_metrics()
        if self.mod == "sqrt":
            self.global_mean_ = np.mean(np.sqrt(X[:, 2]))
            self.std_ = 0
        elif self.mod == "norm":
            self.global_mean_ = np.mean(X[:, 2])
            self.std_ = np.std(X[:, 2])
        else:
            self.global_mean_ = np.mean(X[:, 2])
            self.std_ = 0
        rmse = self._run_sgd(X, X_test)

        return rmse

    def _preprocess_data(self, X, train=True, verbose=True):
        """Maps user and item ids to their indexes.
        Parameters
        ----------
        X : pandas.DataFrame
            Dataset, must have 'u_id' for user ids, 'i_id' for item ids, and
            'rating' column names.
        train : boolean
            Whether or not X is the training set or the validation set.
        Returns
        -------
        X : numpy.array
            Mapped dataset.
        """
        # print('Preprocessing data...\n')
        X = X.copy()

        if train:  # Mappings have to be created)
            user_ids = X["u_id"].unique().tolist()
            item_ids = X["i_id"].unique().tolist()
            n_users = len(user_ids)
            n_items = len(item_ids)
            user_idx = range(n_users)
            item_idx = range(n_items)

            self.user_mapping_ = dict(zip(user_ids, user_idx))
            self.item_mapping_ = dict(zip(item_ids, item_idx))
        X["u_id"] = X["u_id"].map(self.user_mapping_)
        X["i_id"] = X["i_id"].map(self.item_mapping_)
        data=[self.user_mapping_, self.item_mapping_]
        dbfile = open(r"C:\Users\nvtru\Desktop\AI-Recommender\map_user_item", "wb")
        pickle.dump(data,dbfile)

        # Tag validation set unknown users/items with -1 (enables
        # `fast_methods._compute_val_metrics` detecting them)
        X.fillna(-1, inplace=True)
        X["u_id"] = X["u_id"].astype(np.int32)
        X["i_id"] = X["i_id"].astype(np.int32)
        print(X)
        return X[["u_id", "i_id", "rating"]].values

    def _init_metrics(self):
        metrics = np.zeros((self.n_epochs, 3), dtype=float)
        self.metrics_ = pd.DataFrame(metrics, columns=["Loss", "RMSE", "MAE"])

    def _run_sgd(self, X, X_test):
        n_users = len(np.unique(X[:, 0]))
        n_items = len(np.unique(X[:, 1]))

        pu, qi, pu_a, qi_a = _initialization(
            n_users, n_items, self.n_features, self.rand_type, self.deviation
        )
        rmse = 10000.0
        count = 0
        change_rmse = False
        count_lr = 0
        for epoch_ix in range(self.n_epochs):
            start = process_time()
            print("Epoch {}/{}".format(epoch_ix + 1, self.n_epochs), end=": ")
            pu, qi, pu_a, qi_a = _run_epoch(
                X=X,
                pu=pu,
                qi=qi,
                pu_a=pu_a,
                qi_a=qi_a,
                global_mean_=self.global_mean_,
                pen=self.pen,
                lr=self.lr,
                reg=self.reg,
                mod=self.mod,
                std_=self.std_,
            )
            pu_1 = np.array(pd.DataFrame(pu).mean(axis=0))
            qi_1 = np.array(pd.DataFrame(qi).mean(axis=0))
            if self.algorithm == "edulive":
                self.metrics_.loc[epoch_ix, :] = _compute_val_metrics(
                    X=X_test,
                    pu=pu,
                    qi=qi,
                    pu_a=pu_a,
                    qi_a=qi_a,
                    global_mean_=self.global_mean_,
                    mod=self.mod,
                    std_=self.std_,
                )
            if self.algorithm == "edulive_average":
                self.metrics_.loc[epoch_ix, :] = _compute_val_metrics_average(
                    X=X_test,
                    pu=pu,
                    qi=qi,
                    pu_a=pu_a,
                    qi_a=qi_a,
                    global_mean_=self.global_mean_,
                    pu_1=pu_1,
                    qi_1=qi_1,
                    mod=self.mod,
                    std_=self.std_,
                )
            end = process_time()
            print(f"val_loss:", round(self.metrics_.loc[epoch_ix, "Loss"],5), end=" - ")
            print(f"val_rmse:", round(self.metrics_.loc[epoch_ix, "RMSE"],5), end=" - ")
            print(f"val_mae:", round(self.metrics_.loc[epoch_ix, "RMSE"],5), end=" - ")
            print(f"took {end - start:.1f} sec")
            
            # self._on_epoch_end(
            #     start,
            #     self.metrics_.loc[epoch_ix, "Loss"],
            #     self.metrics_.loc[epoch_ix, "RMSE"],
            #     self.metrics_.loc[epoch_ix, "MAE"],
            # )

            val_rmse = self.metrics_.loc[epoch_ix, "RMSE"]
            if val_rmse < rmse - 1e-4:
                count = 0
                count_lr = 0
                rmse = val_rmse
                if self.storefile and (val_rmse < self.min_rmse):
                    change_rmse = True
                    self.min_rmse = val_rmse
                    data = [pu, qi, pu_a, qi_a, self.global_mean_, self.std_]
            else:
                count += 1
                count_lr += 1
            if (count_lr >= 5) and (self.lr > 1e-5):
                self.lr = self.lr / 10
                count_lr = 0
            if count >= 9:
                break

        print(rmse)
        if self.storefile and change_rmse:
            dbfile = open(self.filename, "wb")
            pickle.dump(data, dbfile)
        return rmse

    def _on_epoch_begin(self, epoch_ix):
        """Displays epoch starting log and returns its starting time.

        Parameters
        ----------
        epoch_ix : int
            Epoch index.

        Returns
        -------
        start : float
            Starting time of the current epoch.
        """
        start = time.process_time()
        end = "  | " if epoch_ix < 9 else " | "
        print("Epoch {}/{}".format(epoch_ix + 1, self.n_epochs), end=end)
        return start

    # def _on_epoch_end(
    #     self, start, val_loss=None, val_rmse=None, val_mae=None, process_time=None
    # ):

    #     end = time.process_time()
    #     if val_loss is not None:
    #         print(f"val_loss: {val_loss:.4f}", end=" - ")
    #         print(f"val_rmse: {val_rmse:.4f}", end=" - ")
    #         print(f"val_mae: {val_mae:.4f}", end=" - ")

    #     print(f"took {end - start:.1f} sec")
