
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

    # SIZE = 'small'
    SIZE = 'large'
    file_path = f'hdfs:/user/{netID}/ratings_{SIZE}.csv'
    ratings = spark.read.csv(file_path, header=True, schema='userId INT, movieId INT, rating FLOAT')
    ratings.createOrReplaceTempView('ratings')

    split_size_list = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6]
    N = len(split_size_list)
    for idx in range(N):
        split_size = split_size_list[idx]
        split_data = ratings.sample(split_size, seed=2022)
        split_data.coalesce(1).write.csv(f'hdfs:/user/{netID}/split_no_{idx+1}.csv')



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('extension_data_split').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)