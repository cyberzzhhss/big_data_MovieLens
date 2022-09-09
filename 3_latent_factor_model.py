
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

def main(spark, netID):
    '''
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    # SIZE = 'small'
    SIZE = 'large'
    file_path_training = f'hdfs:/user/{netID}/ratings_{SIZE}_training.csv'
    ratings_training = spark.read.csv(file_path_training, header=True, schema='userId INT, movieId INT, rating FLOAT')
    ratings_training.createOrReplaceTempView('ratings_training')

    # file_path_validation = f'hdfs:/user/{netID}/ratings_{SIZE}_validation.csv'
    # ratings_validation = spark.read.csv(file_path_validation, header=True, schema='userId INT, movieId INT, rating FLOAT')
    # ratings_validation.createOrReplaceTempView('ratings_validation')

    # file_path_test = f'hdfs:/user/{netID}/ratings_{SIZE}_test.csv'
    # ratings_test = spark.read.csv(file_path_test, header=True, schema='userId INT, movieId INT, rating FLOAT')
    # ratings_test.createOrReplaceTempView('ratings_test')
    
    # ------------------------------ Build ALS Model  ----------------------------------------------------
    als = ALS(seed=2022, \
              rank=300,  \
              regParam=0.01, \
              maxIter=10, \
              nonnegative=False, \
              userCol="userId", \
              itemCol="movieId", \
              ratingCol="rating", \
              coldStartStrategy="drop")

    model = als.fit(ratings_training)

    # ------------------------------ RootMeanSquareError Evaluation --------------------------------------

    # evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    # predictionCol="prediction")

    # predictions = model.transform(ratings_training)
    # rmse = evaluator.evaluate(predictions)
    # print("Training Root-mean-square error =   " + str(rmse))

    # predictions2 = model.transform(ratings_validation)
    # rmse2 = evaluator.evaluate(predictions2)
    # print("Validation Root-mean-square error = " + str(rmse2))

    # predictions3 = model.transform(ratings_test)
    # rmse3 = evaluator.evaluate(predictions3)
    # print("Test Root-mean-square error =       " + str(rmse3))

    # ------------------------------ Obtain labels ------------------------------------------------------
    df = ratings_training
    # df = ratings_validation
    # df = ratings_test

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


    # ---------------------------- Unit Tests ------------------------------------------------------------
    
    # print('labels | userId | [[rating, movieId],..]')
    # sliced_df.filter(sliced_df.us1erId==148).show(1,truncate=100)
    
    # print('labels | userId | [movieId,..]')
    # df = sliced_df.select('userId', 'sliced_col.movieId')
    # df.filter(df.userId==148).show(1,truncate=50)
    
    # print('preds | userId | [[movieId, rating],..]')
    # user_subset_recs.filter(user_subset_recs.userId==148).show(1,truncate=100)
    
    # print('preds | userId | [movieId,..]')
    # user_subset_recs.select('userId', 'recommendations.movieId').filter(user_subset_recs.userId==148).show(1,truncate=50)
    
    # print('combined case | userId | label_list | pred_list')
    # final_df.filter(final_df.userId==148).show(1,truncate=50)

    # ------------------------------ Obtain predictionsAndLabels------------------------------------------
    labels = list(final_df.select('label_list').toPandas()['label_list'])
    # l1 = [len(ele) for ele in labels]
    # print(l1)

    preds = list(final_df.select('pred_list').toPandas()['pred_list'])
    # l2 = [len(ele) for ele in preds]
    # print(l2)
    
    predAndLbl = list(zip(preds, labels))

    sc = SparkContext.getOrCreate()
    predictionsAndLabels = sc.parallelize(predAndLbl)
    
    # ------------------------------ Obtain meanAveragePrecision------------------------------------------
    metrics = RankingMetrics(predictionsAndLabels)
    print('MAP')
    print(metrics.meanAveragePrecisionAt(100))
    print("NDCG")
    print(metrics.ndcgAt(100))
    print("Precision")
    print(metrics.precisionAt(100))



    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('latent_factor_model').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)