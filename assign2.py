import time
import copy
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

import DataManager

logname = "Assignment2"

def main():
    setlogger(logname)
    logger = logging.getLogger(logname)

    sampledf, genedf = preprocess(logger)

    train_sampledf = sampledf[sampledf['dataset'] == 'discovery']
    test_sampledf = sampledf[sampledf['dataset'] == 'validation']
    logger.info("Train : {train}, Test : {test}".format(train=len(train_sampledf.index),
                                                        test=len(test_sampledf.index)))

    train_sampleids = train_sampledf['!Sample_geo_accession'].tolist()
    test_sampleids = test_sampledf['!Sample_geo_accession'].tolist()

    train_genedf = genedf[train_sampleids]
    test_genedf = genedf[test_sampleids]
    logger.debug("Train_gene {}, Test_gene {}"
                 .format(train_genedf.shape, test_genedf.shape))

    prob1(train_genedf.transpose(), train_sampledf['Sex'],
          test_genedf.transpose(), test_sampledf['Sex'], logger)

    # prob2(train_genedf.transpose(), train_sampledf['tnm.stage'],
    #       test_genedf.transpose(), test_sampledf['tnm.stage'], logger)
    #
    # prob3(train_genedf.transpose(), train_sampledf['tnm.stage'],
    #      test_genedf.transpose(), test_sampledf['tnm.stage'], logger)

def preprocess(logger):
    logger.info("Loading Data...")
    dm = DataManager.DataManager(logname)

    probedf = dm.probe_table
    probeset_size = len(probedf.index)

    sampledf = dm.sample_table
    sample_size = len(sampledf.index)

    clinicals = filter(lambda x : "!" not in x, list(sampledf.columns))
    clinicals_size = len(clinicals)

    annotable = dm.annotation_table
    probes = list(probedf['ID_REF'])
    annot_probe_gene_dict = dict()
    for p,g in zip(annotable['Probe Set ID'], annotable['Gene Symbol']):
        annot_probe_gene_dict[p] = g

    probe_gene_dict = dict()
    for p in probes:
        probe_gene_dict[p] = annot_probe_gene_dict[p]

    genes = probe_gene_dict.values()
    genes_size = len(set(genes))

    logger.info("\n**PREPROCESS**\nsample size : {s}\n" \
          "probe set size : {p}\n" \
          "clinicals size : {c}\n" \
          "unique gene size : {g}".format(
        s = sample_size, p = probeset_size, c = clinicals_size, g=genes_size
    ))

    genedf = copy.deepcopy(probedf)
    genedf['Gene Symbol'] = map(lambda x : probe_gene_dict[x], genedf['ID_REF'])
    genedf = genedf.groupby('Gene Symbol').mean()

    return sampledf, genedf

def prob1(X, y, test_X, test_y, logger):
    y = map(lambda x : 1 if x=='M' else 0, y)
    test_y = map(lambda x : 1 if x=='M' else 0, test_y)

    f, axarr = plt.subplots(1,2, figsize=(12,6))
    lw = 2

    axarr[0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    axarr[1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    axarr[0].xlabel('False Positive Rate')
    axarr[0].ylabel('True Positive Rate')
    axarr[1].xlabel('False Positive Rate')
    axarr[1].ylabel('True Positive Rate')
    axarr[0].title('Problem 1 ROC Curves - LOO')
    axarr[1].title('Problem 1 ROC Curves - TEST')

    # Logistic Regression - Ridege
    lr = LogisticRegression()
    param_grid = {
        'C':[0.01,0.1,1,10,100]
    }
    grid = GridSearchCV(lr, cv=5, n_jobs=15, param_grid=param_grid)
    t0 = time.time()
    grid.fit(X, y)
    grid_fit = time.time() - t0
    logger.info("Model fitted in %.3f s"
          % grid_fit)

    loo = LeaveOneOut()
    loo_pred = []
    loo_answer = []
    for train_index, test_index  in loo.split(X):
        X_loo_train, X_loo_test = X[train_index], X[test_index]
        y_loo_train, y_loo_test = y[train_index], y[test_index]
        loo_estimator = LogisticRegression(C=grid.best_params_['C'])
        loo_estimator.fit(X_loo_train, y_loo_train)
        loo_pred.append(loo_estimator.predict(X_loo_test))
        loo_answer.append(y_loo_test)

    fpr, tpr, _ = roc_curve(loo_answer, loo_pred)
    roc_auc = auc(fpr, tpr)
    axarr[0].plot(fpr, tpr, color='cornflowerblue',
             lw=lw, label='Logistic Regression - Ridge (AUC = %0.2f)' % roc_auc)
    pred = grid.predict(test_X)
    fpr, tpr, _ = roc_curve(test_y, pred)
    roc_auc = auc(fpr, tpr)
    axarr[1].plot(fpr, tpr, color='cornflowerblue',
             lw=lw, label='Logistic Regression - Ridge (AUC = %0.2f)' % roc_auc)

    # Logistic Regression - Lasso
    lr = LogisticRegression(penalty='l1')
    param_grid = {
        'C':[0.01,0.1,1,10,100]
    }
    grid = GridSearchCV(lr, cv=5, n_jobs=15, param_grid=param_grid)
    t0 = time.time()
    grid.fit(X, y)
    grid_fit = time.time() - t0
    logger.info("Model fitted in %.3f s"
          % grid_fit)

    loo = LeaveOneOut()
    loo_pred = []
    loo_answer = []
    for train_index, test_index  in loo.split(X):
        X_loo_train, X_loo_test = X[train_index], X[test_index]
        y_loo_train, y_loo_test = y[train_index], y[test_index]
        loo_estimator = LogisticRegression(penalty='l1', C=grid.best_params_['C'])
        loo_estimator.fit(X_loo_train, y_loo_train)
        loo_pred.append(loo_estimator.predict(X_loo_test))
        loo_answer.append(y_loo_test)

    fpr, tpr, _ = roc_curve(loo_answer, loo_pred)
    roc_auc = auc(fpr, tpr)
    axarr[0].plot(fpr, tpr, color='cornflowerblue',
             lw=lw, label='Logistic Regression - Ridge (AUC = %0.2f)' % roc_auc)
    pred = grid.predict(test_X)
    fpr, tpr, _ = roc_curve(test_y, pred)
    roc_auc = auc(fpr, tpr)
    axarr[1].plot(fpr, tpr, color='cornflowerblue',
             lw=lw, label='Logistic Regression - Ridge (AUC = %0.2f)' % roc_auc)

    # K Nearest Neighborhood classifier
    lr = KNeighborsClassifier()
    param_grid = {
        'n_neighbors':[3,4,5,6,7]
    }
    grid = GridSearchCV(lr, cv=5, n_jobs=15, param_grid=param_grid)
    t0 = time.time()
    grid.fit(X, y)
    grid_fit = time.time() - t0
    logger.info("Model fitted in %.3f s"
          % grid_fit)

    loo = LeaveOneOut()
    loo_pred = []
    loo_answer = []
    for train_index, test_index  in loo.split(X):
        X_loo_train, X_loo_test = X[train_index], X[test_index]
        y_loo_train, y_loo_test = y[train_index], y[test_index]
        loo_estimator = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'])
        loo_estimator.fit(X_loo_train, y_loo_train)
        loo_pred.append(loo_estimator.predict(X_loo_test))
        loo_answer.append(y_loo_test)

    fpr, tpr, _ = roc_curve(loo_answer, loo_pred)
    roc_auc = auc(fpr, tpr)
    axarr[0].plot(fpr, tpr, color='cornflowerblue',
             lw=lw, label='Logistic Regression - Ridge (AUC = %0.2f)' % roc_auc)
    pred = grid.predict(test_X)
    fpr, tpr, _ = roc_curve(test_y, pred)
    roc_auc = auc(fpr, tpr)

    best_n = str(grid.best_params_['n_neighbors'])
    axarr[1].plot(fpr, tpr, color='darkorange',
             lw=lw, label='KNN[N=%s] (AUC = %0.2f)' % (best_n,roc_auc))

    plt.legend(loc="lower right")
    plt.savefig("results/ass2_prob1.png")

def prob2(X, y, test_X, test_y, logger):
    # Logistic Regression - Ridege
    lr = ElasticNet(l1_ratio=0)
    param_grid = {
    }
    grid = GridSearchCV(lr, cv=5, n_jobs=15, param_grid=param_grid)
    t0 = time.time()
    grid.fit(X, y)
    grid_fit = time.time() - t0
    logger.info("Model fitted in %.3f s"
          % grid_fit)

    loo = LeaveOneOut()
    loo_pred = []
    loo_answer = []
    for train_index, test_index  in loo.split(X):
        X_loo_train, X_loo_test = X[train_index], X[test_index]
        y_loo_train, y_loo_test = y[train_index], y[test_index]
        loo_estimator = ElasticNet(l1_ratio=0)
        loo_estimator.fit(X_loo_train, y_loo_train)
        loo_pred.append(loo_estimator.predict(X_loo_test))
        loo_answer.append(y_loo_test)

    loo_rmse = (sum(np.square(np.array(loo_pred) - np.array(test_y)))/len(loo_pred))**0.5
    pred = grid.predict(test_X)
    rmse = (sum(np.square(np.array(pred) - np.array(test_y)))/len(pred))**0.5
    logger.info("LOO)Linear Regression - Ridge : %s"%str(loo_rmse))
    logger.info("TEST)Linear Regression - Ridge : %s"%str(rmse))


    # Logistic Regression - Lasso
    lr = ElasticNet(l1_ratio=1)
    param_grid = {
    }
    grid = GridSearchCV(lr, cv=5, n_jobs=15, param_grid=param_grid)
    t0 = time.time()
    grid.fit(X, y)
    grid_fit = time.time() - t0
    logger.info("Model fitted in %.3f s"
          % grid_fit)
    loo = LeaveOneOut()
    loo_pred = []
    loo_answer = []
    for train_index, test_index  in loo.split(X):
        X_loo_train, X_loo_test = X[train_index], X[test_index]
        y_loo_train, y_loo_test = y[train_index], y[test_index]
        loo_estimator = ElasticNet(l1_ratio=1)
        loo_estimator.fit(X_loo_train, y_loo_train)
        loo_pred.append(loo_estimator.predict(X_loo_test))
        loo_answer.append(y_loo_test)

    loo_rmse = (sum(np.square(np.array(loo_pred) - np.array(test_y)))/len(loo_pred))**0.5
    pred = grid.predict(test_X)
    rmse = (sum(np.square(np.array(pred) - np.array(test_y)))/len(pred))**0.5
    logger.info("LOO)Linear Regression - Lasso : %s"%str(loo_rmse))
    logger.info("TEST)Linear Regression - Lasso : %s"%str(rmse))


    lr = DecisionTreeRegressor()
    param_grid = {
    }
    grid = GridSearchCV(lr, cv=5, n_jobs=15, param_grid=param_grid)
    t0 = time.time()
    grid.fit(X, y)
    grid_fit = time.time() - t0
    logger.info("Model fitted in %.3f s"
          % grid_fit)
    loo = LeaveOneOut()
    loo_pred = []
    loo_answer = []
    for train_index, test_index  in loo.split(X):
        X_loo_train, X_loo_test = X[train_index], X[test_index]
        y_loo_train, y_loo_test = y[train_index], y[test_index]
        loo_estimator = DecisionTreeRegressor()
        loo_estimator.fit(X_loo_train, y_loo_train)
        loo_pred.append(loo_estimator.predict(X_loo_test))
        loo_answer.append(y_loo_test)

    loo_rmse = (sum(np.square(np.array(loo_pred) - np.array(test_y)))/len(loo_pred))**0.5
    pred = grid.predict(test_X)
    rmse = (sum(np.square(np.array(pred) - np.array(test_y)))/len(pred))**0.5
    logger.info("LOO)Decision Tree Regression : %s"%str(loo_rmse))
    logger.info("TEST)Decision Tree Regression : %s"%str(rmse))

def prob3(X, y, test_X, test_y, logger):
    y = map(lambda x : 0 if x in [0,1,2] else 1, y)
    test_y = map(lambda x : 0 if x in [0,1,2] else 1, test_y)
    f, axarr = plt.subplots(1,2, figsize=(8,4))
    lw = 2

    axarr[0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    axarr[1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    axarr[0].xlabel('False Positive Rate')
    axarr[0].ylabel('True Positive Rate')
    axarr[1].xlabel('False Positive Rate')
    axarr[1].ylabel('True Positive Rate')
    axarr[0].title('Problem 3 ROC Curves - LOO')
    axarr[1].title('Problem 3 ROC Curves - TEST')

    # Logistic Regression - Ridege
    lr = LogisticRegression()
    param_grid = {
        'C':[0.01,0.1,1,10,100]
    }
    grid = GridSearchCV(lr, cv=5, n_jobs=15, param_grid=param_grid)
    t0 = time.time()
    grid.fit(X, y)
    grid_fit = time.time() - t0
    logger.info("Model fitted in %.3f s"
          % grid_fit)

    loo = LeaveOneOut()
    loo_pred = []
    loo_answer = []
    for train_index, test_index  in loo.split(X):
        X_loo_train, X_loo_test = X[train_index], X[test_index]
        y_loo_train, y_loo_test = y[train_index], y[test_index]
        loo_estimator = LogisticRegression(C=grid.best_params_['C'])
        loo_estimator.fit(X_loo_train, y_loo_train)
        loo_pred.append(loo_estimator.predict(X_loo_test))
        loo_answer.append(y_loo_test)

    fpr, tpr, _ = roc_curve(loo_answer, loo_pred)
    roc_auc = auc(fpr, tpr)
    axarr[0].plot(fpr, tpr, color='cornflowerblue',
             lw=lw, label='Logistic Regression - Ridge (AUC = %0.2f)' % roc_auc)
    pred = grid.predict(test_X)
    fpr, tpr, _ = roc_curve(test_y, pred)
    roc_auc = auc(fpr, tpr)
    axarr[1].plot(fpr, tpr, color='cornflowerblue',
             lw=lw, label='Logistic Regression - Ridge (AUC = %0.2f)' % roc_auc)

    # Logistic Regression - Lasso
    lr = LogisticRegression(penalty='l1')
    param_grid = {
        'C':[0.01,0.1,1,10,100]
    }
    grid = GridSearchCV(lr, cv=5, n_jobs=15, param_grid=param_grid)
    t0 = time.time()
    grid.fit(X, y)
    grid_fit = time.time() - t0
    logger.info("Model fitted in %.3f s"
          % grid_fit)

    loo = LeaveOneOut()
    loo_pred = []
    loo_answer = []
    for train_index, test_index  in loo.split(X):
        X_loo_train, X_loo_test = X[train_index], X[test_index]
        y_loo_train, y_loo_test = y[train_index], y[test_index]
        loo_estimator = LogisticRegression(penalty='l1', C=grid.best_params_['C'])
        loo_estimator.fit(X_loo_train, y_loo_train)
        loo_pred.append(loo_estimator.predict(X_loo_test))
        loo_answer.append(y_loo_test)

    fpr, tpr, _ = roc_curve(loo_answer, loo_pred)
    roc_auc = auc(fpr, tpr)
    axarr[0].plot(fpr, tpr, color='cornflowerblue',
             lw=lw, label='Logistic Regression - Ridge (AUC = %0.2f)' % roc_auc)
    pred = grid.predict(test_X)
    fpr, tpr, _ = roc_curve(test_y, pred)
    roc_auc = auc(fpr, tpr)
    axarr[1].plot(fpr, tpr, color='cornflowerblue',
             lw=lw, label='Logistic Regression - Ridge (AUC = %0.2f)' % roc_auc)

    # K Nearest Neighborhood classifier
    lr = KNeighborsClassifier()
    param_grid = {
        'n_neighbors':[3,4,5,6,7]
    }
    grid = GridSearchCV(lr, cv=5, n_jobs=15, param_grid=param_grid)
    t0 = time.time()
    grid.fit(X, y)
    grid_fit = time.time() - t0
    logger.info("Model fitted in %.3f s"
          % grid_fit)

    loo = LeaveOneOut()
    loo_pred = []
    loo_answer = []
    for train_index, test_index  in loo.split(X):
        X_loo_train, X_loo_test = X[train_index], X[test_index]
        y_loo_train, y_loo_test = y[train_index], y[test_index]
        loo_estimator = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'])
        loo_estimator.fit(X_loo_train, y_loo_train)
        loo_pred.append(loo_estimator.predict(X_loo_test))
        loo_answer.append(y_loo_test)

    fpr, tpr, _ = roc_curve(loo_answer, loo_pred)
    roc_auc = auc(fpr, tpr)
    axarr[0].plot(fpr, tpr, color='cornflowerblue',
             lw=lw, label='Logistic Regression - Ridge (AUC = %0.2f)' % roc_auc)
    pred = grid.predict(test_X)
    fpr, tpr, _ = roc_curve(test_y, pred)
    roc_auc = auc(fpr, tpr)

    best_n = str(grid.best_params_['n_neighbors'])
    axarr[1].plot(fpr, tpr, color='darkorange',
             lw=lw, label='KNN[N=%s] (AUC = %0.2f)' % (best_n,roc_auc))

    plt.legend(loc="lower right")
    plt.savefig("results/ass2_prob3.png")

def setlogger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
    fileHandler = logging.FileHandler("logs/{}.log".format(logname))
    streamHandler = logging.StreamHandler()

    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logger

if __name__ == "__main__":
    main()
