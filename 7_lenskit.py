#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:47:36 2022

@author: joepei
"""
from lenskit.datasets import ML100K
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit import topn
import pandas as pd
from lenskit.metrics.predict import user_metric, rmse
import time

split1 = pd.read_csv('split_no_1.csv', header = None)
split2 = pd.read_csv('split_no_2.csv', header = None)
split3 = pd.read_csv('split_no_3.csv', header = None)
split4 = pd.read_csv('split_no_4.csv', header = None)
split5 = pd.read_csv('split_no_5.csv', header = None)
split6 = pd.read_csv('split_no_6.csv', header = None)


split1.columns = ['user', 'item', 'rating']
split2.columns = ['user', 'item', 'rating']
split3.columns = ['user', 'item', 'rating']
split4.columns = ['user', 'item', 'rating']
split5.columns = ['user', 'item', 'rating']
split6.columns = ['user', 'item', 'rating']


timeTable = []
precisionTable = []
percentages = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6]

def eval(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    start_time = time.time()
    fittable.fit(train)
    print("--- %s seconds ---" % (time.time() - start_time))
    users = train.user.unique()
    # now we run the recommender
    recs = batch.recommend(fittable, users, 100, n_jobs = 1)
    # add the algorithm name for analyzability
    recs['Algorithm'] = aname
    return recs

for i in range(4,7):
    print("split: %2d" % (i))
    data = globals()["split"+str(i)]
    algo_als = als.BiasedMF(20, iterations = 10, reg = 0.01)
    #algo_li = knn.ItemItem(20)
    all_recs = []
    train_data = []
    for train, test in xf.partition_users(data[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2)):
        train_data.append(train)
        #all_recs.append(eval('ItemItem', algo_ii, train, test))
        all_recs.append(eval('ALS', algo_als, train, test))
    
    all_recs = pd.concat(all_recs, ignore_index=True)
    #test_data = pd.concat(test_data, ignore_index=True)
    train_data = pd.concat(train_data, ignore_index = True)
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.precision)
    results = rla.compute(all_recs, train_data)
    print(results.groupby('Algorithm').precision.mean())


