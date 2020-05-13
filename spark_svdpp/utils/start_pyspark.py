from pyspark.sql import SparkSession


def yarn_client():
    spark = (
        SparkSession.builder
        .master('yarn_client')
        .appName('demo')
        .config('spark.yarn.jars', 'hdfs://kronos1:9000/spark_jars/*')
        .config('spark.driver.memory', '64g')
        .config('spark.executor.memory', '64g')
        .config('spark.driver.maxResultSize', '10G')  # driver預設只吃1GB的回傳序列化物件
        .getOrCreate()
    )
    return spark


def local():
    spark = (
        SparkSession.builder
        .master('local[*]')
        .appName('local')
        .config('spark.driver.memory', '64g')
        .config('spark.executor.cores', 10)
        .config('spark.executor.memory', '64g')
        .config('spark.driver.maxResultSize', '10G')  # driver預設只吃1GB的回傳序列化物件
        .getOrCreate()
    )
    return spark
