import numpy as np
from .dataread import DataRead as dr
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import randint as sp_uniform
import pandas as pd
from sklearn.metrics import roc_auc_score
from .configurations import configurations as cf
from .performance import Performance as pf



data = dr.data_read()


def parameters_list(conf):

    """
    This function return the candidate parameters matrix for the classifier/model.
    :param conf: Number of configurations of parameter for classifier
    :return: Parameter matrix
    """
    train_size = len(data.train_x)  # Size of training set
    parameters = []
    configurations = conf  # Number of Configurations
    algorithm_param_numbers = 5
    list_parameters = np.zeros(algorithm_param_numbers)  # List for sorting the parameters of algorithm
    for c in range(0, configurations):
        params_iterate = []  # Parameters list for each iteration
        clf = RandomForestClassifier()  # Chosen classifier for prediction
        param_dist = {  # Choose discreet and continuous values for parameters randomly
            "n_estimators": sp_randint(1, 500),
            "max_depth": sp_uniform(0.1, 1),
            "max_features": sp_randint(1, 11),
            "min_samples_split": sp_randint(2, 20),
            "min_samples_leaf": sp_randint(1, 11)
        }
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist)  # Randomized search on hyper parameters
        train_x = dr.DataRead.train_x()
        y = dr.DataRead.train_y()
        random_forest_model = random_search.fit(train_x, y)
        for key in random_search.best_params_:
            params_iterate.append(random_search.best_params_[key])
        list_parameters = np.vstack((list_parameters, params_iterate))
    return list_parameters


def CSVT_main_loop(fold, wstop, configuration, alpha, beta):
    """
     This function is a main loop of the cross validation algorithm which selects top configurations
    :param wstop: 
    :param beta:
    :param alpha:
    :param configuration:
    :param fold:
    :return: none
    """
    train_x = dr.DataRead.train_x()
    y = dr.DataRead.train_y()
    performance_mean = np.empty([configuration, fold])  # Matrix for storing Mean Performance
    performance_mean[:] = np.NAN  # ?????????????????????????????????
    matrix_trace = np.zeros(configuration)
    isActive = np.ones(configuration)  # Active Configuration

    train_size = len(data.train_x)
    n = (train_size + 1) / fold
    print("n", n)  # Initialize subset increment
    pp_matrix = np.zeros(train_size - 2)  # Pointwise performance matrix
    top_configurations = []  # for storing the top configurations
    score = []  # ?????????????????????????????????
    Ty = np.zeros(configuration)
    print(isActive)
    list_parameters = parameters_list(configuration)
    for fd in range(1, fold + 1):  # To find the top performing configurations for fold fd
        print("fold Number", fd)  # Total Number of folds for cross validation
        performance_matrix = []
        for c in range(0, configuration):
            if isActive[c] == 1:
                K = sum(isActive)
                ind1 = (fd - 1) * int(n)
                ind2 = (fd * int(n)) - 2
                print("index 1......", ind1)
                print("index 2....", ind2)
                x = train_x.values[ind1:ind2]  # Range of dataset of train in current fold
                z = y[ind1:ind2]  # Range of dataset of prediction in current fold
                test_CV = train_x.drop(train_x.index[[ind1, ind2]])  # Rest of the data for testing
                v = pd.DataFrame(y)
                y_test = v.drop(v.index[[ind1, ind2]])  # prediction of test data
                myarray = np.array(parameters_list[c]).tolist()
                clf = RandomForestClassifier(n_estimators=myarray[0], max_depth=myarray[1],
                                                                          max_features=myarray[2],
                                                                          min_samples_split=myarray[3],
                                                                          min_samples_leaf=myarray[4])
                random_forest_model = clf.fit(x, z)
                reds = clf.predict_proba(test_CV)  # Predict probabilities
                reds = np.array(reds[:, 1]).tolist()
                print("y_test", y_test)
                print("red", reds)
                y_test = np.array(y_test)
                roc_score = roc_auc_score(y_test, reds)
                print("roc_score", roc_score)
                performance_matrix = np.append(performance_matrix, roc_score)
                performance_mean[c, fd - 1] = roc_score  # Mean performance of each configuration
                for l in range(0, len(reds)):  # conversion of XGboost prediction into binary form
                    if reds[l] > 0.5:
                        reds[l] = 1
                    else:
                        reds[l] = 0
                print("#" * 10)
                c = c - 1
                pp_matrix = np.vstack((pp_matrix, reds))

        pp_matrix = np.delete(pp_matrix, 0, 0)  # pointwise peroformance matrix
        top_configurations = cf.top_configurations(pp_matrix, performance_matrix, alpha,
                                                    K)  # Find the top configurations
        A = np.where(isActive == 1)
        print(A[0])
        Ty[A[0]] = top_configurations
        matrix_trace = np.vstack((matrix_trace, Ty))
        # Configurations are column-wise and folds are Row-wise                                      #Top configurations are "1" in columns

        print("isActive", isActive)
        for z in range(0, len(matrix_trace[0])):
            T = cf.is_flop_configuration(matrix_trace[:, z], fd, fold, beta,
                                         alpha)  # Checking each configuration whether its Flop or not
            if T is not None:  # D-Active Flop Configuration
                isActive[z] = 0
        print("is ative", isActive)
        isActive_index = np.where(isActive == 1)  # Slection the index of configurations which are not flop
        isActive_index = np.array(isActive_index)
        print("isActive_index[0]", isActive_index[0])
        trace_matrix = np.delete(matrix_trace, 0, 0)
        trace_matrix = trace_matrix.T
        print("trace_matrix\n", trace_matrix)
        p = pf.similar_performance(trace_matrix[isActive_index[0], (f - wstop + 1):f], alpha)
        if p <= alpha:  # checks whether all remaining configurations performed equally well in the past
            break
    Final_asnwer = pf.select_winner(performance_mean, isActive, wstop, f)
    print("Final_answer", Final_asnwer)