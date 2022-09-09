#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:36:04 2022

@author: joepei
"""

from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
import pandas as pd 
from lightfm.data import Dataset
import time


#Data was pre-splitted using Spark for control of results and reproducibility

split1 = pd.read_csv('split_no_1.csv', header = None)
split2 = pd.read_csv('split_no_2.csv', header = None)
split3 = pd.read_csv('split_no_3.csv', header = None)
split4 = pd.read_csv('split_no_4.csv', header = None)
split5 = pd.read_csv('split_no_5.csv', header = None)
split6 = pd.read_csv('split_no_6.csv', header = None)


split1.columns = ['userId', 'movieId', 'rating']
split2.columns = ['userId', 'movieId', 'rating']
split3.columns = ['userId', 'movieId', 'rating']
split4.columns = ['userId', 'movieId', 'rating']
split5.columns = ['userId', 'movieId', 'rating']
split6.columns = ['userId', 'movieId', 'rating']


timeTable = []
precisionTable = []
percentages = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6]
for i in range(1,7):
    print("split: %2d" % (i))
    dataset = Dataset()
    data = globals()["split"+str(i)]
    print(data)
    dataset.fit(data["userId"], data["movieId"])
    num_users, num_items = dataset.interactions_shape()
    (interactions, weights) = dataset.build_interactions(((int(r[0]), int(r[1]), r[2]) for i, r in data.iterrows()))
    #Because we are trying to optimize precision@k, we decided to use 'warp' loss 
    #This is the Weighted Approximate-Rank Pairwise loss
    #Maximises the rank of positive examples by repeatedly sampling negative examples until rank violating one is found.
    model = LightFM(loss='warp', no_components=300, user_alpha= 0.01)
    start_time = time.time()
    model.fit(interactions, sample_weight = weights, epochs = 10, num_threads = 1)
    print("--- %s seconds ---" % (time.time() - start_time))
    train_precision = precision_at_k(model, interactions, k=100).mean()
    print("---train precision = %10.8f ---" % (train_precision))
    
    