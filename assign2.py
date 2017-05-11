import time

import copy
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

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


    prob1(train_genedf.transpose(), train_sampledf['Sex'], test_genedf.transpose(), test_sampledf['Sex'])
    prob2()
    prob3()

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

def prob1(X, y, test_X, test_y):
    print X, len(y)
    train_size =len(y)

    pipe = Pipeline([
        ('reduce_dim', PCA()),
        ('classify', SVC())
    ])

    N_FEATURES_OPTIONS = [100, 300, 1000, 3000]
    #N_FEATURES_OPTIONS = [100]
    C_OPTIONS = [1e0, 1e1, 1e2, 1e3]
    #C_OPTIONS = [1e0]
    GAMMA_OPTIONS = np.logspace(-2, 2, 5)
    #GAMMA_OPTIONS = [0.03]
    param_grid = {
        'reduce_dim': [PCA(iterated_power=7), NMF()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS,
        'classify__gamma' : GAMMA_OPTIONS
    }
    reducer_labels = ['PCA', 'NMF']

    grid = GridSearchCV(pipe, cv=5, n_jobs=15, param_grid=param_grid)
    t0 = time.time()

    grid.fit(X, y)
    grid_fit = time.time() - t0
    logger.info("PipeLine fitted in %.3f s"
          % grid_fit)

    cv_df = pd.DataFrame(grid.cv_results_)
    cv_df.to_csv("cv_result.csv")

    #grid_ratio = grid.best_estimator_.support_.shape[0] / train_size
    #print("Support vector ratio: %.3f" % grid_ratio)

    # mean_scores = np.array(grid.cv_results_['mean_test_score'])
    # # scores are in the order of param_grid iteration, which is alphabetical
    # mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
    # # select score for best C
    # mean_scores = mean_scores.max(axis=0)
    # bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
    #                (len(reducer_labels) + 1) + .5)

    pass

def prob2():
    pass

def prob3():
    pass

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
