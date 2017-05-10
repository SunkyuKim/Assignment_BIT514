from __future__ import division

from scipy import stats

import copy
import logging
import numpy as np
import progressbar
import scipy.stats
import seaborn as sns
from sklearn import linear_model

import DataManager

"""
_probset : 54675"
_sample : 585
_clinical variables : 33, (34 with "dependancy sample")
_unique genes?? - paper read...
"""

logname = "result2_prob3"
logger = None # Loading in Main

def main():
    setlogger(logname)
    logger = logging.getLogger(logname)

    logger.info("Loading Data...")
    dm = DataManager.DataManager(logname)

    logger.info("***Prob 1")

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

    logger.info("\nsample size : {s}\n" \
          "probe set size : {p}\n" \
          "clinicals size : {c}\n" \
          "unique gene size : {g}".format(
        s = sample_size, p = probeset_size, c = clinicals_size, g=genes_size
    ))

    genedf = copy.deepcopy(probedf)
    genedf['Gene Symbol'] = map(lambda x : probe_gene_dict[x], genedf['ID_REF'])
    genedf = genedf.groupby('Gene Symbol').mean()

    def prob2():
        logger.info("***Prob 2")
        male_sampledf = sampledf[sampledf['Sex']=='M']
        female_sampledf = sampledf[sampledf['Sex']=='F']
        logger.info("Male samples : {}, Female samples : {}".format(
            len(male_sampledf.index), len(female_sampledf.index)
        ))

        male_sampleids = male_sampledf['!Sample_geo_accession'].tolist()
        female_sampleids = female_sampledf['!Sample_geo_accession'].tolist()

        # Probeset
        male_probedf = probedf[male_sampleids]
        female_probedf = probedf[female_sampleids]
        logger.debug("Male_prob {}, Female_prob {}"
                     .format(male_probedf.shape, female_probedf.shape))

        logger.info("T-test for every prob into M/F...")
        probe_ref_list = probedf['ID_REF'].tolist()
        proberef_pvalue = dict()

        bar = progressbar.ProgressBar()
        for i in bar(range(len(probe_ref_list))):
            male_values = male_probedf.iloc[i].tolist()
            female_values = female_probedf.iloc[i].tolist()
            t, p = scipy.stats.ttest_ind(male_values, female_values)
            proberef_pvalue[probe_ref_list[i]] = p

        top_probes = sorted(proberef_pvalue, key=proberef_pvalue.get)
        for i,key in enumerate(top_probes[:10]):
             logger.info("{}) {} {}".format(i,key,proberef_pvalue[key]))

        count = 0
        for k in proberef_pvalue:
            if proberef_pvalue[k] < 0.001:
                count += 1
        logger.info("Probes whose p-value is lower than 0.001 : {}".format(count))

        # Gene
        male_genedf = genedf[male_sampleids]
        female_genedf = genedf[female_sampleids]
        logger.debug("Male_gene {}, Female_gene {}"
                     .format(male_genedf.shape, female_genedf.shape))

        logger.info("T-test for every gene into M/F...")
        gene_ref_list = genedf.index.tolist()
        generef_pvalue = dict()
        bar = progressbar.ProgressBar()
        for i in bar(range(len(gene_ref_list))):
            male_values = male_genedf.iloc[i].tolist()
            female_values = female_genedf.iloc[i].tolist()
            t, p = scipy.stats.ttest_ind(male_values, female_values)
            generef_pvalue[gene_ref_list[i]] = p

        top_genes = sorted(generef_pvalue, key=generef_pvalue.get)
        for i,key in enumerate(top_genes[:10]):
             logger.info("{}) {} {}".format(i,key,generef_pvalue[key]))

        count = 0
        for k in generef_pvalue:
            if generef_pvalue[k] < 0.001:
                count += 1
        logger.info("Genes whose p-value is lower than 0.001 : {}".format(count))

        top200genes = top_genes[:200]

        D = [genedf.loc[g].tolist() for g in top200genes]
        D = np.array(D)
        logger.debug(D.shape)
        sns.clustermap(D,figsize=(20, 10),cmap='hot')
        sns.plt.savefig('results/prob2_heatmap.png')

    def prob3():
        logger.info("**Prob 3")

        logger.info("T-test and Permutation test for every gene into KRAS M/WT...")
        krasM_df = sampledf[sampledf['kras.mutation']=="M"]
        krasWT_df = sampledf[sampledf['kras.mutation']=="WT"]
        logger.info("kras mutated sample : {}, wild type sampe : {}".format(
            len(krasM_df.index), len(krasWT_df.index)
        ))
        krasM_sampleids = krasM_df['!Sample_geo_accession'].tolist()
        krasWT_sampleids = krasWT_df['!Sample_geo_accession'].tolist()

        krasM_genedf = genedf[krasM_sampleids]
        krasWT_genedf = genedf[krasWT_sampleids]
        gene_ref_list = genedf.index.tolist()
        generef_pvalue_prob3ttest = dict()
        generef_pvalue_prob3permutationtest = dict()

        bar = progressbar.ProgressBar()
        for i in bar(range(len(gene_ref_list))):
            m_values = krasM_genedf.iloc[i].tolist()
            wt_values = krasWT_genedf.iloc[i].tolist()
            t,p = scipy.stats.ttest_ind(m_values, wt_values)
            generef_pvalue_prob3ttest[gene_ref_list[i]] = p
            # p_ptest = exact_montecarlo_perm_test(m_values, wt_values, 1000)
            p_ptest = t_montecarlo_perm_test(m_values, wt_values, 10000, t)

            generef_pvalue_prob3permutationtest[gene_ref_list[i]] = p_ptest

        top_genes_prob3ttest = sorted(generef_pvalue_prob3ttest, key=generef_pvalue_prob3ttest.get)
        top_genes_prob3ptest = sorted(generef_pvalue_prob3permutationtest, key=generef_pvalue_prob3permutationtest.get)
        for i in range(10):
            k1 = top_genes_prob3ttest[i]
            k2 = top_genes_prob3ptest[i]
            logger.info("{}) {} {} {} {}".format(
                i,k1,generef_pvalue_prob3ttest[k1],k2,generef_pvalue_prob3permutationtest[k2]
            ))

        count = 0
        for k in generef_pvalue_prob3ttest:
            if generef_pvalue_prob3ttest[k] < 0.001:
                count += 1
        logger.info("[t-test]Genes whose p-value is lower than 0.001 : {}".format(count))

        count = 0
        for k in generef_pvalue_prob3permutationtest:
            if generef_pvalue_prob3permutationtest[k] < 0.001:
                count += 1
        logger.info("[permutation-test]Genes whose p-value is lower than 0.001 : {}".format(count))

    def prob4():
        logger.info("**Prob 4")

        logger.info("Regression test for every gene...")

        stagedf = sampledf[sampledf['tnm.stage']<5]
        # sampleids = stagedf['!Sample_geo_accession'].tolist()
        sampleids = stagedf['!Sample_geo_accession'].tolist()
        # sample_stages = stagedf['tnm.stage'].tolist()
        sample_stages = stagedf['tnm.stage'].tolist()

        print sampledf.groupby('tnm.stage').count()['!Sample_geo_accession']

        y_values = []
        for v in sample_stages:
            if v == 0:
                y_values.append(1)
            else:
                y_values.append(v)
        y_np = np.array([[v] for v in y_values])

        prob4_genedf = genedf[sampleids]
        gene_ref_list = genedf.index.tolist()
        prob4_generef_pvalue_regr = dict()
        bar = progressbar.ProgressBar()
        for i in bar(range(len(gene_ref_list))):
            x_values = prob4_genedf.iloc[i].tolist()
            # x_np = np.array([[1,v] for v in x_values])
            # regr = LinearRegression_pvalue()
            # regr.fit(x_np, y_np)
            # prob4_generef_pvalue_regr[gene_ref_list[i]] = regr.p
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_values,y_values)
            prob4_generef_pvalue_regr[gene_ref_list[i]] = p_value

        prob4_top_genes = sorted(prob4_generef_pvalue_regr, key=prob4_generef_pvalue_regr.get)
        for i,key in enumerate(prob4_top_genes[:10]):
             logger.info("{}) {} {}".format(i,key,prob4_generef_pvalue_regr[key]))

        count = 0
        for k in prob4_generef_pvalue_regr:
            if prob4_generef_pvalue_regr[k] < 0.001:
                count += 1
        logger.info("[Regr]Genes whose p-value is lower than 0.001 : {}".format(count))

        logger.info("T-test for every gene into Benign/Malignant...")

        benign_ids = sampledf[sampledf['tnm.stage'] < 3]['!Sample_geo_accession'].tolist()
        malig_ids = sampledf[sampledf['tnm.stage'] >= 3]['!Sample_geo_accession'].tolist()

        logger.info("Benign sample : {}, Malignant sampe : {}".format(
            len(benign_ids), len(malig_ids)
        ))

        benign_genedf = genedf[benign_ids]
        malig_genedf = genedf[malig_ids]

        gene_ref_list = genedf.index.tolist()
        prob4_generef_pvalue = dict()

        bar = progressbar.ProgressBar()
        for i in bar(range(len(gene_ref_list))):
            benign_values = benign_genedf.iloc[i].tolist()
            malig_values = malig_genedf.iloc[i].tolist()
            t, p = scipy.stats.ttest_ind(benign_values, malig_values)
            prob4_generef_pvalue[gene_ref_list[i]] = p

        prob4_top_genes = sorted(prob4_generef_pvalue, key=prob4_generef_pvalue.get)
        for i,key in enumerate(prob4_top_genes[:10]):
             logger.info("{}) {} {}".format(i,key,prob4_generef_pvalue[key]))

        count = 0
        for k in prob4_generef_pvalue:
            if prob4_generef_pvalue[k] < 0.001:
                count += 1
        logger.info("[t-test]Genes whose p-value is lower than 0.001 : {}".format(count))

        returndict = {'sig':[], 'unsig':[]}
        for k in prob4_top_genes:
            if prob4_generef_pvalue[k] < 0.001:
                returndict['sig'].append(k)
            else:
                returndict['unsig'].append(k)
        return returndict

    def prob5(sig_unsig_genes_dict):
        logger.info("**Prob 5")

        fr = open("res/raw/c5.all.v6.0.symbols.gmt")

        genesetref_list = []
        genesets = []
        for l in fr:
            tokens = l.split("\t")
            setref = tokens[0]
            genes = tokens[2:]
            genesetref_list.append(setref)
            genesets.append(genes)

        genesetref_pvalue_dict = dict()

        bar = progressbar.ProgressBar()
        for i in bar(range(len(genesets))):
            gs = genesets[i]
            a = len([v for v in sig_unsig_genes_dict['sig'] if v in gs])
            b = len([v for v in sig_unsig_genes_dict['sig'] if v not in gs])
            c = len([v for v in sig_unsig_genes_dict['unsig'] if v in gs])
            d = len([v for v in sig_unsig_genes_dict['unsig'] if v not in gs])
            oddsratio, pvalue = stats.fisher_exact([[a, b], [c, d]])
            genesetref_pvalue_dict[genesetref_list[i]] = pvalue

        prob5_top_geneset = sorted(genesetref_pvalue_dict, key=genesetref_pvalue_dict.get)
        for i,key in enumerate(prob5_top_geneset[:10]):
             logger.info("{}) {} {}".format(i,key,genesetref_pvalue_dict[key]))

        count = 0
        for k in genesetref_pvalue_dict:
            if genesetref_pvalue_dict[k] < 0.001:
                count += 1
        logger.info("[t-test]Genes whose p-value is lower than 0.001 : {}".format(count))


    # prob2()
    prob3()
    # sig_unsig_genes_dict = prob4()
    # prob5(sig_unsig_genes_dict)

def t_montecarlo_perm_test(xs, ys, nmc, ref):
    n, k = len(xs), 0

    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        t_z,p_z = scipy.stats.ttest_ind(zs[:n], zs[n:])
        if ref < t_z:
            k+=1
    return float(k) / float(nmc)

def exact_montecarlo_perm_test(xs, ys, nmc):
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))

    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / float(nmc)


class LinearRegression_pvalue(linear_model.LinearRegression):

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(LinearRegression_pvalue, self)\
                .__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression_pvalue, self).fit(X, y, n_jobs)

        rse  = np.sqrt(np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0]-2))
        se = rse**2 / float(np.sum((X[:,1]-np.mean(X[:,1]))**2,axis=0))

        self.t = self.coef_[0,1] / float(np.sqrt(se))
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), X.shape[0] - 2))
        return self

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

if __name__ == '__main__':
    main()