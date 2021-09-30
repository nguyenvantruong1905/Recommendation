from SVD import SVD_Andrew
from SVD import SVD_Edulive
from SVD import SVD_Koren
import pandas as pd

###########################################################################################
# path_file_train = (r"C:\Users\nvtru\Desktop\AI-Recommender\movieslen_data\train.txt")
# path_file_test = (r"C:\Users\nvtru\Desktop\AI-Recommender\movieslen_data\test1.txt")
# train = pd.read_csv(path_file_train, sep=",",
#                    names=["u_id", "i_id", "rating", "timestamps"])
# test = pd.read_csv(path_file_test, sep=",",
#                    names=["u_id", "i_id", "rating", "timestamps"])
path_file_train = (r"E:\recall\train.txt")
path_file_test = (r"E:\recall\test.txt")
train = pd.read_csv(path_file_train, sep=",",
                names=["u_id", "i_id", "rating","timestamps"])
test = pd.read_csv(path_file_test, sep=",",
                names=["u_id", "i_id", "rating","timestamps"])
############################################################################################
svd_andrew = SVD_Andrew(lr=1e-3, reg=0.43214549207553254, n_epochs=300, n_features=64, algorithm='andrew')
############################################################################################
svd_edulive = SVD_Edulive(lr=1e-3, reg=0.006059592463013812, pen= 24.759991720712186, n_features=3, algorithm= 'edulive')
############################################################################################
svd_koren = SVD_Koren(lr=0.001, reg=0.001, n_epochs=200, n_features=100, algorithm= 'koren')
############################################################################################
svd_andrew.fit(train,test)
