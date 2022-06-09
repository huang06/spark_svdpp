from pyspark.rdd import RDD

from ..dataset import Dataset


def test_get_samples(spark):
    path = 'hdfs:///svdpp/dataset_sample.parquet'
    dataset = Dataset()
    dataset.load_dataset(spark=spark, path=path)

    samples = dataset.get_samples(k=10).collect()
    for par in samples:
        assert len(par[0]) == len(par[1])
        assert len(par[0]) == 10


def test_load_dataset(spark):
    path = 'hdfs:///svdpp/dataset_sample.parquet'
    dataset = Dataset()
    dataset.load_dataset(spark=spark, path=path)

    df = spark.read.parquet(path).toPandas()
    assert df['i'].nunique() == dataset.n_items
    assert df['u'].nunique() == dataset.n_users

    assert isinstance(dataset.ur, dict)
    assert len(dataset.ur) == dataset.n_users
    assert isinstance(dataset.item_id_index, dict)
    assert len(dataset.item_id_index) == dataset.n_items
    assert isinstance(dataset.user_id_index, dict)
    assert len(dataset.user_id_index) == dataset.n_users
    assert isinstance(dataset.item_index_id, dict)
    assert len(dataset.item_index_id) == dataset.n_items
    assert isinstance(dataset.user_index_id, dict)
    assert len(dataset.user_index_id) == dataset.n_users
    assert isinstance(dataset.train, RDD)
