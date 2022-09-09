
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
from pyspark.sql.functions import *
from pyspark.mllib.evaluation import *
from pyspark import SparkContext
# from pyspark.sql import functions as F

def main(spark, netID):
    '''Main routine for Lab Solutions
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

    file_path_test = f'hdfs:/user/{netID}/ratings_{SIZE}_test.csv'
    ratings_test = spark.read.csv(file_path_test, header=True, schema='userId INT, movieId INT, rating FLOAT')
    ratings_test.createOrReplaceTempView('ratings_test')

    # small is 32
    # large is 18895
    if SIZE == 'small':
        hyperparam = 32
    else:
        hyperparam = 18895

    model = spark.sql(f'SELECT movieId, AVG(ratings_training.rating) AS rating_avg \
                       FROM ratings_training \
                       GROUP BY movieId \
                       HAVING COUNT(ratings_training.rating) > {hyperparam} \
                       ORDER BY rating_avg DESC\
                       LIMIT 100') # evaluation is based on top 100 recommended movies

    raw_prediction = list(model.select('movieId').toPandas()['movieId'])
    cnt = ratings_training.select('userId').distinct().count() # this variable is a int
    predictions = [raw_prediction for _ in range(cnt)]


    # df = ratings_training
    # df = ratings_validation
    df = ratings_test

    sorted_df = df.groupBy('userId').agg(sort_array(collect_list(struct(col('rating'),col('movieId'))),asc=False).alias('sorted_col'))
    # # index starts at 1 in slice, amazingly!!!
    sliced_df = sorted_df.withColumn('sliced_col', slice(col('sorted_col'),1,100)).drop('sorted_col') 
    labels_df = sliced_df.select(sliced_df.userId, sliced_df.sliced_col.movieId.alias('movieId_list'))
    # labels_df.show(1, False)
    labels = list(labels_df.select('movieId_list').toPandas()['movieId_list'])

    # print(len(labels[0]))
    # print(len(labels[122]))
    # print('labels userId length:', len(labels))
    # print('cnt: ', cnt)

    predAndLbl = list(zip(predictions, labels))
    sc = SparkContext.getOrCreate()
    predictionsAndLabels = sc.parallelize(predAndLbl)
    metrics = RankingMetrics(predictionsAndLabels)
    print('MAP')
    print(metrics.meanAveragePrecisionAt(100))
    print("NDCG")
    print(metrics.ndcgAt(100))
    print("Precision")
    print(metrics.precisionAt(100))

    # print(labels[0])
    # print(labels[122])
    # detupled_df.show(10)
    # print(detupled_df.count())


    # save model
    # model.coalesce(1).write.csv(f'hdfs:/user/{netID}/baseline_model_{SIZE}.csv')




    
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)