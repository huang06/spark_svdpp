#!/usr/bin/env bash

zip -r spark_svdpp.zip spark_svdpp -x "*.parquet" -x "*.csv"

spark-submit \
--master 'yarn' \
--driver-memory '50g' \
--num-executors '32' \
--executor-memory '24g' \
--conf spark.driver.maxResultSize='20g' \
--conf spark.executor.cores='4' \
--conf spark.yarn.jars='hdfs:///spark_jars/*' \
--py-files spark_svdpp.zip \
examples/svdpp_example.py 8
