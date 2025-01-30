import numpy as np
import pandas as pd
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Build tree from splitting

# each tree
# (i) for each feature: find_split
# (ii) Mapbypartition(find_split)

def feature_split(dataset, feature_array):
    '''
    Input:
    partition: a pyspark dataframe partition to be called by foreachPartition,
    feature_array: a broadcasted feature array for the tree that is intiialized earlier on
    '''
    #define schema
    schema = StructType([
        StructField("feature", IntegerType(), True),
        StructField("split_value", FloatType(), True),
        StructField("info_gain", FloatType(), True),
    ])
    feature_df = spark.createDataFrame([], schema)

    # for each feature array, get a split and append the dataframe
    for feature_index in feature_array:

        # find split
        feature_split = new_split(dataset, feature_index)

        #add feature
        feature_df = feature_df.union(feature_split)

    return feature_df




#old
#for each partition, find the best split given a feature index
# x_train & y_train are RDD of x and y variables respectively
# feature_index represents the feature being trained
# parent_data_count is a global sum broadcasted at the start of computation

def find_split(joined_df, feature_index):

    #split x_train & y_train

    y_train = joined_df.select(joined_df.columns[-1])
    split_data = joined_df.select(col(joined_df.columns[feature_index]).alias("feature"),col(joined_df.columns[-1]).alias("y"))

    #init variables
    parent_entropy = class_entropy(y_train)
    parent_data_count = split_data.count()
    schema = StructType([
        StructField("feature", IntegerType(), True),
        StructField("index", IntegerType(), True),
        StructField("split_value", FloatType(), True),
        StructField("info_gain", FloatType(), True),
    ])
    best_IG = 0
    best_split = 0
    best_idx = 0


    #for each point in x_train for feature_index, compute information gain
    for (idx,split_col) in enumerate(split_data.rdd.distinct().collect()):

        #get split value
        split_value = split_col[0]

        #threshold
        split_data = split_data.withColumn("x_child_left", when(split_data[0] <= split_value,True).otherwise(False))
        split_data = split_data.withColumn("x_child_right", when(split_data[0] > split_value,True).otherwise(False))

        #join and get
        #joined_df = x_threshold.join(y_train,col('index'))
        y_child_left = split_data.filter(col("x_child_left")).select(col('y'))
        y_child_right = split_data.filter(col("x_child_right")).select(col('y'))

        #calculate entropy
        entropy_left = class_entropy(y_child_left)
        entropy_right = class_entropy(y_child_right)

        #calculate Information Gain
        num_left, num_right = y_child_left.count(), y_child_right.count()
        if num_left != 0 and num_right != 0:
            left_weighted = entropy_left * (num_left / parent_data_count)
            right_weighted = entropy_right * (num_right/parent_data_count)
            IG =  float(parent_entropy  - left_weighted - right_weighted)

            #check best
            if IG > best_IG:
                best_IG = IG
                best_split = split_value
                best_idx = idx

    IG_df = spark.createDataFrame([(feature_index,best_idx,float(best_split),float(best_IG))], schema)

    return IG_df

