
from SVD import SVD_Edulive
import pandas as pd
import numpy as np 

###########################################################################################
path_file_train = (r"C:\Users\nvtru\ai-machine\movieslen_data\train.txt")
path_file_test = (r"C:\Users\nvtru\ai-machine\movieslen_data\test1.txt")
train = pd.read_csv(path_file_train, sep=",",
                   names=["u_id", "i_id", "rating", "timestamps"])
test = pd.read_csv(path_file_test, sep=",",
                   names=["u_id", "i_id", "rating", "timestamps"])
time_exam_train = np.random.choice(30, 80000)
time_exam_test = np.random.choice(30, 20000)
action_exam_train = np.random.choice(50, 80000)
action_exam_test = np.random.choice(50, 20000)
train['time_exam'] = pd.Series(time_exam_train,index=train.index)
train['action_exam'] = pd.Series(action_exam_train,index=train.index)
test['time_exam'] = pd.Series(time_exam_test,index=test.index)
test['action_exam'] = pd.Series(action_exam_test,index=test.index)

# algorithm = 'edulive' : chay thuat toan edulive
# algorithm = 'edulive_average' :  chay thuat toan edulive_average    
svd_edulive = SVD_Edulive(lr=0.005, reg=0.0001, pen= 0.0001, n_features=3, n_epochs=20,  lambda1= 0.001, lambda2= 0.001, beta=0.001, alpha= 0.001)
############################################################################################

svd_edulive.fit(train,test)
