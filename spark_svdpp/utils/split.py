import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as f


spark = SparkSession.builder.master('local[*]').getOrCreate()
sc = spark.sparkContext
seed = 9999

current_dir = os.getcwd()
dataset = spark.read.parquet(f"file://{current_dir}/dataset.parquet")

dataset.cache().count()
print('set uids')
uids = (
    dataset.select('u').distinct()
    .rdd.flatMap(lambda e: e).takeSample(False, 1000, seed)
)
print('get train set')
dataset_test = dataset.filter(f.col('u').isin(uids))
print('get test set')
dataset_train = dataset.filter(f.col('u').isin(uids) == False)
print('write to local')
dataset_train.write.parquet(f"file://{current_dir}/dataset_train.parquet")
dataset_test.write.parquet(f"file://{current_dir}/dataset_test.parquet")
