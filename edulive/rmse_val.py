from SVD import algorithm_svd_andrew
from SVD import algorithm_svd_koren
from SVD import algorithm_svd_edulive


user_mapping_,item_mapping_= _preprocess_train(train)
user_item_rating_from_val_set = _preprocess_val(val,user_mapping_,item_mapping_)
#################################################################################
# algorithm = koren : tinh cac loss, rmse, process_time theo koren
# algorithm = koren_average : tinh cac loss, rmse, process_time theo koren_average
# algorithm = koren_bubi_average : tinh cac loss, rmse, process_time theo koren_bubi_average

def evaluate_koren(file, algorithm='koren'):
    list_params = pickle.load(file)
    bu=list_params[0]
    bi=list_params[1]
    pu=list_params[2]
    qi=list_params[3]
    global_mean = list_params[4]
    if algorithm == 'koren':
        return algorithm_svd_koren._compute_val_metrics(user_item_rating_from_val_set, bu, bi, pu, qi, global_mean)
    if algorithm == 'koren_average':
        return algorithm_svd_koren._compute_val_metrics_average(user_item_rating_from_val_set, bu, bi, pu, qi, global_mean)
    if algorithm == 'koren_bubi_average':
        return algorithm_svd_koren._compute_val_metrics_bubi_average(user_item_rating_from_val_set, bu, bi, pu, qi, global_mean)
    if algorithm == 'koren_bypass':
        return algorithm_svd_koren._compute_val_metrics_bypass(user_item_rating_from_val_set, bu, bi, pu, qi, global_mean)

#################################################################################

def evaluate_andrew(file, algorithm='andrew'):
    list_params = pickle.load(file)
    pu=list_params[0]
    qi=list_params[1]
    if algorithm == 'andrew':
        return algorithm_svd_andrew._compute_val_metrics(user_item_rating_from_val_set, pu, qi)
    if algorithm == 'andrew_bypass':
        return algorithm_svd_andrew._compute_val_metrics_bypass(user_item_rating_from_val_set, pu, qi)
#################################################################################
# algorithm = 'edulive' tinh cac loss, rmse, process_time theo edulive
# algorithm = 'edulive_average' tinh cac loss, rmse, process_time theo edulive_average

def avaluate_edulive(file, algorithm='edulive'):
    list_params = pickle.load(file)
    pu=list_params[0]
    qi=list_params[1]
    pu_a=list_params[2]
    qi_a=list_params[3]
    global_mean = list_params[4]
    if algorithm == 'edulive':
        return algorithm_svd_edulive._compute_val_metrics(user_item_rating_from_val_set,pu, qi, pu_a, qi_a, global_mean)
    if algorithm == 'edulive_average':
        return algorithm_svd_edulive._compute_val_metrics_average(user_item_rating_from_val_set,pu, qi, pu_a, qi_a, global_mean)
    if algorithm == 'edulive_bypass':
        return algorithm_svd_edulive._compute_val_metrics_bypass(user_item_rating_from_val_set,pu, qi, pu_a, qi_a, global_mean)

file = open("/home/truonv/Desktop/AI-Recommender/project/test_Koren_params",'rb')
print(evaluate_koren(file,algorithm='koren'))