# BIG DATA

# Project


# Overview

In the project, we applied the tools we learned to solve a realistic, large-scale applied problem.
Specifically, we built and evaluate a collaborative-filter based recommender system. 

## The data set

In this project, we'll use the [MovieLens](https://grouplens.org/datasets/movielens/latest/) datasets collected by 
> F. Maxwell Harper and Joseph A. Konstan. 2015. 
> The MovieLens Datasets: History and Context. 
> ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

The data is hosted in NYU's HPC environment under `/scratch/work/courses/DSGA1004-2021/movielens`.

Two versions of the dataset are provided: a small sample (`ml-latest-small`, 9000 movies and 600 users) and a larger sample (`ml-latest`, 58000 movies and 280000 users).
Each version of the data contains rating and tag interactions, and the larger sample includes "tag genome" data for each movie, which we may consider as additional features beyond
the collaborative filter.
Each version of the data includes a README.txt file which explains the contents and structure of the data which are stored in CSV files.

I strongly recommend to thoroughly read through the dataset documentation before beginning, and make note of the documented differences between the smaller and larger datasets.
Knowing these differences in advance saves we many headaches when it comes time to scale up.

## Basic recommender system 

1.  As a first step, we need to partition the rating data into training, validation, and test samples as discussed in lecture.
    I recommend writing a script do this in advance, and saving the partitioned data for future use.
    This reduces the complexity of our experiment code down the line, and make it easier to generate alternative splits if we want to measure the stability of our
    implementation.

2.  Before implementing a sophisticated model, we should begin with a popularity baseline model as discussed in class.
    This should be simple enough to implement with some basic dataframe computations.
    Evaluate our popularity baseline (see below) before moving on to the enxt step.

3.  our recommendation model should use Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.
    Be sure to thoroughly read through the documentation on the [pyspark.ml.recommendation module](https://spark.apache.org/docs/3.0.1/ml-collaborative-filtering.html) before getting started.
    This model has some hyper-parameters that we should tune to optimize performance on the validation set, notably: 
      - the *rank* (dimension) of the latent factors, and
      - the regularization parameter.

### Evaluation

Once we are able to make predictions—either from the popularity baseline or the latent factor model—we need to evaluate accuracy on the validation and test data.
Scores for validation and test should both be reported in our write-up.
Evaluations should be based on predictions of the top 100 items for each user, and report the ranking metrics provided by spark.
Refer to the [ranking metrics](https://spark.apache.org/docs/3.0.1/mllib-evaluation-metrics.html#ranking-systems) section of the Spark documentation for more details.

The choice of evaluation criteria for hyper-parameter tuning is up to we, as is the range of hyper-parameters we consider, but be sure to document our choices in the final report.
As a general rule, we should explore ranges of each hyper-parameter that are sufficiently large to produce observable differences in our evaluation score.

If we like, we may also use additional software implementations of recommendation or ranking metric evaluations, but be sure to cite any additional software we use in the project.


### Using the cluster

Please be considerate of our fellow classmates!
The Peel cluster is a limited, shared resource. 
Make sure that our code is properly implemented and works efficiently. 
If too many people run inefficient code simultaneously, it can slow down the entire cluster for everyone.
