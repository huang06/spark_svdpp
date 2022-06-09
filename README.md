# PySpark implementation of SVD++ for Top-N Recommendation

![pyspark-flow](img/pyspark-flow.png)

## Getting Started

### Prerequisites

You need to install *Apache Hadoop* and *Apache Spark* on every nodes of the cluster.

#### Install Hadoop

```bash
tar zxvf hadoop-3.y.z.tgz
ln -s /your/hadoop/path/hadoop-3.x.z /your/hadoop/path/hadoop
```

#### Install Spark

```bash
tar zxvf spark-2.y.z-bin-hadoop2.7.tgz
ln -s /your/spark/path/spark-2.y.z /your/spark/path/spark
```

### Installing

#### Clone the repository

```bash
git clone git@bitbucket.org:citomhuang/spark_svdpp.git
```

#### Create the Python environment

```bash
cd spark_svdpp

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt  # development purpose
```

#### Run the tests

```bash
pytest spark_svdpp
```

## Run a example

```bash
./yarn-client.sh
```

## References

- [Factorization Meets the Neighborhood: A Multifaceted Collaborative Filtering Model. Yehuda Koren, KDD’08](https://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf)
- [Spark: Cluster Computing with Working Sets](https://www.usenix.org/legacy/event/hotcloud10/tech/full_papers/Zaharia.pdf)
- [Scaling Collaborative Filtering with PySpark](https://engineeringblog.yelp.com/2018/05/scaling-collaborative-filtering-with-pyspark.html)
- [Running Spark on YARN](https://spark.apache.org/docs/latest/running-on-yarn.html)
- [NicolasHug/Surprise](https://github.com/NicolasHug/Surprise)
