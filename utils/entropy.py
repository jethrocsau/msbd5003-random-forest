import numpy as np
import pandas as pd


def class_entropy(df):
    # Example entropy calculation for binary classification
    col_name = "y"
    counts = df.groupBy(col_name).count()
    total = df.count()
    return counts.withColumn("prob", F.col("count") / total).select(
        F.sum(-F.col("prob") * F.log2(F.col("prob"))).alias("entropy")
    ).first()["entropy"]

def prob(df):
    # Example entropy calculation for binary classification
    count = np.count_nonzero(df)
    if count == 0:
        return float(0)
    else:
        total = len(df)
        prob = np.divide(count,total)
        return float(prob)


