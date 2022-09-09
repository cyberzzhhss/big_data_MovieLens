
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:
    $ spark-submit baseline.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass
import sys
# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.mllib.evaluation import *
from pyspark import SparkContext
import time

def main(spark, netID):
    '''
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    precision_list = []
    duration_list = []

    # the split size is consistent with that of extension_data_split
    split_size_list = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6]
    N = len(split_size_list)
    for idx in range(N):
        file_path = f'hdfs:/user/{netID}/split_no_{idx+1}.csv'
        ratings = spark.read.csv(file_path, header=True, schema='userId INT, movieId INT, rating FLOAT')
        ratings.createOrReplaceTempView('ratings')
        
        # ------------------------------ Build ALS Model  ----------------------------------------------------
        # the best parameter from our parameter searching regime
        start_time = time.time()

        als = ALS(seed=2022, \
                rank=300,  \
                regParam=0.01, \
                maxIter=10, \
                nonnegative=False, \
                userCol="userId", \
                itemCol="movieId", \
                ratingCol="rating", \
                coldStartStrategy="drop")

        model = als.fit(ratings)

        end_time = time.time()
        duration = end_time - start_time

        duration_list.append(duration)

        # ------------------------------ Obtain labels -----------------------------------------
        df = ratings

        sorted_df = df.groupBy('userId').agg(sort_array(collect_list(struct(col('rating'),col('movieId'))),asc=False) \
                                        .alias('sorted_col'))

        ## index starts at 1 in slice, amazingly!!!
        sliced_df = sorted_df.withColumn('sliced_col', slice(col('sorted_col'),1,100)).drop('sorted_col') 

        # ------------------------------ Obtain Predictions --------------------------------------------------
        # Evaluate the model by computing the RMSE on the test data
        user_subset = sliced_df.select('userId')
        user_subset_recs = model.recommendForUserSubset(user_subset, 100)
        
        final_df = sliced_df.join(user_subset_recs, user_subset_recs.userId == sliced_df.userId, 'inner') \
                            .select(sliced_df.userId, \
                                    sliced_df.sliced_col.movieId.alias('label_list'), \
                                    user_subset_recs.recommendations.movieId.alias('pred_list'))

        # ------------------------------ Obtain predictionsAndLabels------------------------------------------
        labels = list(final_df.select('label_list').toPandas()['label_list'])
        preds = list(final_df.select('pred_list').toPandas()['pred_list'])
        predAndLbl = list(zip(preds, labels))
        sc = SparkContext.getOrCreate()
        predictionsAndLabels = sc.parallelize(predAndLbl)

                
        # ------------------------------ Obtain meanAveragePrecision------------------------------------------
        metrics = RankingMetrics(predictionsAndLabels)
        print("Precision")
        print(prec)
        print('Duration')
        print(duration)
        prec = metrics.precisionAt(100)
        precision_list.append(prec)

    print('precision list')
    print(precision_list)

    print('duration list')
    print(duration_list)

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('extension_als').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)