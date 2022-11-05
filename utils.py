# -*- coding: utf-8 -*-
import logging, argparse, os
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
from sklearn import preprocessing

def set_logging(path, filename):
    if not os.path.exists(path):
        os.mkdir(path)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                        datefmt='%d %b %Y %H:%M:%S',
                        filename=path + '/' + filename + '.log',
                        filemode='w+')
    logger = logging.getLogger('Decorr')
    if not logger.handlers or len(logger.handlers) == 0:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger

def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='given a model file name')
    args = parser.parse_args()
    return args

def weighted_stat(x, weights):
    n, p = x.shape
    statModel = DescrStatsW(data=x, weights=weights)
    corr_mat = statModel.corrcoef # column * column 
    mean_pairwise_correlation = (np.sum(np.abs(corr_mat))-p)/p/(p-1) # exclude 1; p * (p - 1) correlation remain
    #Compute Condition Number
    eigenvalue = np.linalg.eigvals(corr_mat)
    max_eig = eigenvalue.max()
    min_eig = eigenvalue.min()
    condition_index = max_eig/eigenvalue
    condition_number = np.linalg.cond(corr_mat) # cond(A) = ||A|| * ||A^(-1)||; default by 2-norm
    w_stat = {}
    w_stat['corrcoef'] = corr_mat
    w_stat['mean_corr'] = mean_pairwise_correlation
    w_stat['min_eig'] = min_eig
    w_stat['CI'] = condition_index
    w_stat['CN'] = condition_number
    return w_stat

def predict(X, beta):
    return np.matmul(X, beta)

def cal_prediction_error(y_true, y_predict, loss_type = 'rmse'):
    if loss_type == 'rmse':
        return np.sqrt(np.mean(np.square(y_true.squeeze() - y_predict.squeeze())))
    elif loss_type == 'mse':
        return np.mean(np.square(y_true.squeeze() - y_predict.squeeze()))

def cal_estimation_error(beta_true, beta_predict, loss_type = 'ae'):
    if loss_type == 'rmse':
        return np.sqrt(np.mean(np.square(beta_true - beta_predict)))
    elif loss_type == 'mae':
        return np.mean(np.abs(beta_true - beta_predict))
    elif loss_type == 'mse':
        return np.mean(np.square(beta_true - beta_predict))
    elif loss_type == 'ae':
        return np.sum(np.abs(beta_true - beta_predict))
