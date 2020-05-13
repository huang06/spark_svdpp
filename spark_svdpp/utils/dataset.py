"""Dataset"""
import numpy as np
import pyspark
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType


DEFAULT_ITEM_COL = 'i'
DEFAULT_USER_COL = 'u'
DEFAULT_RATING_COL = 'r'
DEFAULT_NUM_PARTITIONS = 5


class Dataset:
    """Dataset class."""
    def __init__(self):
        self.item_id_index = None
        self.user_id_index = None
        self.item_index_id = None
        self.user_index_id = None
        self.train = None
        self.n_items = None
        self.n_users = None
        self.ur = None

    def get_samples(self, k: int):
        def get_samples_local(par, k):
            assert len(par[0]) == len(par[1])
            assert k >= 1
            n_elements = len(par[0])
            n_samples = min(n_elements, k)
            indices = np.random.choice(range(n_elements), n_samples).tolist()
            return (
                [par[0][idx] for idx in indices], [par[1][idx] for idx in indices]
            )
        return self.train.map(lambda par: get_samples_local(par, k))

    def load_dataset(self,
                     spark: pyspark.sql.SparkSession,
                     path: str,
                     item_col: str = DEFAULT_ITEM_COL,
                     user_col: str = DEFAULT_USER_COL,
                     rating_col: str = DEFAULT_RATING_COL,
                     n_pars: int = DEFAULT_NUM_PARTITIONS) -> None:
        """Load dataset."""
        # check parameters
        if not isinstance(spark, pyspark.sql.SparkSession):
            raise TypeError
        if not isinstance(path, str):
            raise TypeError
        if not isinstance(n_pars, int) or n_pars <= 0:
            raise ValueError

        # TODO split data into training data and validation data

        sdf1 = spark.read.parquet(path)
        colnames = set(sdf1.columns)

        # check column names
        for colname in (item_col, user_col, rating_col):
            if colname not in colnames:
                raise ValueError(f'column {colname} not in data')

        # make mapper
        item_index_id = {iidx: iid for iidx, iid in enumerate(sdf1.select(item_col).distinct().rdd.map(lambda x: x[0]).collect())}
        user_index_id = {uidx: uid for uidx, uid in enumerate(sdf1.select(user_col).distinct().rdd.map(lambda x: x[0]).collect())}
        item_id_index = {iid: iidx for iidx, iid in item_index_id.items()}
        user_id_index = {uid: uidx for uidx, uid in user_index_id.items()}
        mapforwardI = F.udf(item_id_index.get, IntegerType())
        mapforwardU = F.udf(user_id_index.get, IntegerType())
        sdf2 = (
            sdf1
            .withColumn(item_col, mapforwardI(sdf1[item_col]))
            .withColumn(user_col, mapforwardU(sdf1[user_col]))
        )
        ur = (
            sdf2
            .groupby(user_col)
            .agg(F.collect_set(item_col)).rdd.collectAsMap()
        )  # return a dict: key denotes user_index, value denotes item_indices user interacted.

        # TODO should we eliminate the redundant map-processes ?
        # sdf2.map(lambda row: (row[item_col], row[user_col], row[rating_col])).repartition(5).glom()
        rdd = sdf2.rdd.repartition(n_pars).cache()
        rdd_r = rdd.map(lambda row: row[rating_col]).glom()
        rdd_iu = rdd.map(lambda row: (row[item_col], row[user_col])).glom()
        rdd_iur = rdd_iu.zip(rdd_r)

        self.ur = ur
        self.n_items = len(item_index_id)
        self.n_users = len(user_index_id)
        self.item_id_index = item_id_index
        self.user_id_index = user_id_index
        self.item_index_id = item_index_id
        self.user_index_id = user_index_id
        self.train = rdd_iur
