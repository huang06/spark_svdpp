# PySpark implementation of SVD++ for Top-N Recommendation

![pyspark-flow](img/pyspark-flow.png)

## Prerequisites

You need to install *Apache Hadoop* and *Apache Spark* on every nodes of the cluster.

### Install Hadoop

```bash
tar zxvf hadoop-3.y.z.tgz
ln -s /your/hadoop/path/hadoop-3.x.z /your/hadoop/path/hadoop
```

### Install Spark

```bash
tar zxvf spark-2.y.z-bin-hadoop2.7.tgz
ln -s /your/spark/path/spark-2.y.z /your/spark/path/spark
```

## Getting Started

### Create the Python environment

```bash
make python
```

### Run tests

```bash
make test
```

### Run example

```bash
make example
```

## References

- [Factorization Meets the Neighborhood: A Multifaceted Collaborative Filtering Model. Yehuda Koren, KDD’08](https://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf)
- [Spark: Cluster Computing with Working Sets](https://www.usenix.org/legacy/event/hotcloud10/tech/full_papers/Zaharia.pdf)
- [Scaling Collaborative Filtering with PySpark](https://engineeringblog.yelp.com/2018/05/scaling-collaborative-filtering-with-pyspark.html)
- [Running Spark on YARN](https://spark.apache.org/docs/latest/running-on-yarn.html)
- [NicolasHug/Surprise](https://github.com/NicolasHug/Surprise)
