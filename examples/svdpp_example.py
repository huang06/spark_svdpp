import sys
from pyspark.sql import SparkSession
from spark_svdpp.algos.svdpp import run


n_pars = int(sys.argv[1])
spark = SparkSession.builder.getOrCreate()
output_data_path = 'hdfs:///svdpp/output.parquet'
input_data_path = 'hdfs:///svdpp/dataset_train.parquet'

run(spark=spark, n_pars=n_pars,
    input_data_path=input_data_path,
    output_data_path=output_data_path)
