
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
import math

def main(spark, netID):
    '''
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    SIZE = 'small'
    # SIZE = 'large'

    file_path = f'hdfs:/user/{netID}/ratings_{SIZE}.csv'
    ratings = spark.read.csv(file_path, header=True, schema='userId INT, movieId INT, rating FLOAT')
    ratings.createOrReplaceTempView('ratings')

    file_path_training = f'hdfs:/user/{netID}/ratings_{SIZE}_training.csv'
    ratings_training = spark.read.csv(file_path_training, header=True, schema='userId INT, movieId INT, rating FLOAT')
    ratings_training.createOrReplaceTempView('ratings_training')

    file_path_validation = f'hdfs:/user/{netID}/ratings_{SIZE}_validation.csv'
    ratings_validation = spark.read.csv(file_path_validation, header=True, schema='userId INT, movieId INT, rating FLOAT')
    ratings_validation.createOrReplaceTempView('ratings_validation')

    file_path_test = f'hdfs:/user/{netID}/ratings_{SIZE}_test.csv'
    ratings_test = spark.read.csv(file_path_test, header=True, schema='userId INT, movieId INT, rating FLOAT')
    ratings_test.createOrReplaceTempView('ratings_test')


    # ---------------- understanding the split ------------------------------------
    print('Distinct users in the INNER JOIN between training and validation')
    query = spark.sql('SELECT COUNT(DISTINCT ratings_validation.userId)\
                       FROM ratings_validation \
                       INNER JOIN ratings_training ON ratings_training.userId = ratings_validation.userId')
    query.show()

    print('Distinct users in the INNER JOIN between training and test')
    query2 = spark.sql('SELECT COUNT(DISTINCT ratings_test.userId)\
                       FROM ratings_test \
                       INNER JOIN ratings_training ON ratings_training.userId = ratings_test.userId')
    query2.show()

    print('training total num of userId  ', ratings_training.select('userId').distinct().count())
    print('validation total num of userId', ratings_validation.select('userId').distinct().count())
    print('test total num of userId      ', ratings_test.select('userId').distinct().count())

    # ---------------- understanding the split size ------------------------------------

    print('unsplit total num of records   ', ratings.select('userId').count())
    print('training total num of records  ', ratings_training.select('userId').count())
    print('validation total num of records', ratings_validation.select('userId').count())
    print('test total num of records      ', ratings_test.select('userId').count())

    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('data_exploration').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)