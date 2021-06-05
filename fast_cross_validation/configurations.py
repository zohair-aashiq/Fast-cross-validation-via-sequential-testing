import numpy as np
import statsmodels.stats.api as ap
import math

from pandas.tests.indexing.test_indexing import setitem
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


class configurations:

    def top_configurations(P1, mean_performance, alpha, K):

        """
        :param mean_performance:
        :param alpha:
        :param K:
        :return:
        """
        sum_1 = []
        sort_pre = []
        Y = P1.shape
        print("Mean Performance", mean_performance)  # mean performance of all configuration
        t = np.argsort(mean_performance)
        P_pp = t[::-1]
        sorted_list = np.zeros(Y[1])
        for s in range(0, len(P_pp)):
            sorted_list = np.vstack((sorted_list, P1[P_pp[s]]))  # Sort Pp according to the mean performance
        sorted_list = np.delete(sorted_list, 0, 0)
        sorted_list = np.matrix(np.array(sorted_list))
        alpha_2 = alpha / K  # K is number of Active configuration
        for k in range(1, len(sorted_list)):
            p = ap.cochrans_q(sorted_list[0:k, :])  # Cochrans_q test for finding the top configuration
            if p[1] <= alpha_2:
                break
        k = k + 1
        for x in range(0, k):
            sorted_list[x] = 1  # Top configuration as a 1
        for y in range(k, len(sorted_list)):
            sorted_list[y] = 0
        for s in range(0, len(sorted_list)):  # Rests are zeros
            sum_1.append(sorted_list[s].max())
        sort_pre = np.zeros(len(sum_1))
        for b, c in zip(P_pp, sum_1):
            setitem(sort_pre, b, c)  # Configurations back to their original sequence
        return sort_pre

    def is_flop_configuration(T, s, S, beta=0.1, alpha=0.01):

        """
        :param s:
        :param S:
        :param beta:
        :param alpha:
        :return:
        """
        pi_0 = 0.05
        pi_1 = 0.0
        g = 1 / float(S)
        alpha_l = 0.01
        beta_l = 0.1
        pi_1 = (((1 - beta_l) / alpha_l) ** g) / 2
        num = math.log(beta_l / (1 - alpha_l))
        dum = math.log(pi_1 / pi_0) - math.log((1 - pi_1) / (1 - pi_0))
        num_1 = math.log((1 - pi_0) / (1 - pi_1))
        dum_1 = math.log(pi_1 / pi_0) - math.log((1 - pi_1) / (1 - pi_0))
        a = num / dum
        b = num_1 / dum_1
        if sum(T) <= (a + b * s):
            return T