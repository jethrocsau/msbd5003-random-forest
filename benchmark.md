# Benchmarking

-   Test among different split method
    -   ID3 - DF
    -   C3.4 - DF
    -   CART - DF

|     | train_size | k_features | n_estimators | percent_of_categorical       |
| --- | ---------- | ---------- | ------------ | ---------------------------- |
| 1   | 10_000     | 10         | 100          | 1                            |
| 2   | 1000       | 10         | 100          | 0 (all continuous numerical) |
| 3   | 100_000    | 20         | 100          | 0.5                          |
| 4   | 100_000    | 20         | 10000        | 0.5                          |

(The table is expected to be sorted in ascending order by runtime, not confident)

-   Assume all other parameters to be fixed
-   For memory analysis, reason the code to derive instead of actual benchmarking
-   For runtime performance analysis, benchmark the time required for each combination

# Algo to test

-   sklearn random forest (non parallel)
-   spark with weight used to determine subsets
-   spark with subset passed between workers

# May test

-   Num of partition
