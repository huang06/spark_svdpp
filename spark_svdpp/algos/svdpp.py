"""An implementation of SVD++ in PySpark."""
import time

import numpy as np
import pyspark
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from ..utils.dataset import Dataset

DEFAULT_LOGLEVEL = 'ERROR'


def get_u_impl_fdb(n_users: int, n_factors: int, ur: dict[int, list[int]], yj: np.ndarray) -> np.ndarray:
    """Compute user implicit feedback."""
    # check parameters
    if not isinstance(n_users, int) and n_users <= 0:
        raise ValueError
    if not isinstance(n_factors, int) and n_factors <= 0:
        raise ValueError
    if not isinstance(ur, dict):
        raise TypeError
    for u_index, i_indices in ur.items():
        if not isinstance(u_index, int) and u_index >= 0:
            raise TypeError
        if not isinstance(i_indices, list):
            raise TypeError
        if not all(isinstance(i_index, int) and i_index >= 0 for i_index in i_indices):
            raise ValueError
    if not isinstance(yj, np.ndarray):
        raise TypeError

    u_impl_fdb = np.zeros((n_users, n_factors), np.float32)
    for user_index, item_indices in ur.items():
        sqrt_iu = np.sqrt(len(item_indices))
        for j in item_indices:
            u_impl_fdb[user_index] += yj[j] / sqrt_iu
    if u_impl_fdb.shape != (n_users, n_factors):
        raise AssertionError
    return u_impl_fdb


def map_sgd(
    loss_list: list[tuple[int, int, float]],
    bi: np.ndarray,
    bu: np.ndarray,
    qi: np.ndarray,
    pu: np.ndarray,
    yj: np.ndarray,
    u_impl_fdb: np.ndarray,
    lr_bi: float | int,
    lr_bu: float | int,
    lr_qi: float | int,
    lr_pu: float | int,
    lr_yj: float | int,
    reg_bi: float | int,
    reg_bu: float | int,
    reg_qi: float | int,
    reg_pu: float | int,
    reg_yj: float | int,
    ur: dict[int, list[int]],
) -> tuple[
    tuple[str, np.ndarray],
    tuple[str, np.ndarray],
    tuple[str, np.ndarray],
    tuple[str, np.ndarray],
    tuple[str, np.ndarray],
]:
    """Update paramaters."""
    for i, u, loss in loss_list:
        # update parameters
        Iu = ur[u]
        sqrt_iu = np.sqrt(len(Iu))
        # update bu[u]
        bu[u] += lr_bu * (loss - reg_bu * bu[u])
        # update bi[i]
        bi[i] += lr_bi * (loss - reg_bi * bi[i])
        # update pu[u]
        pu[u] += lr_pu * (loss * qi[i] - reg_pu * pu[u])
        # update qi[i]
        qi[i] += lr_qi * (loss * (pu[u] + u_impl_fdb[u]) - reg_qi * qi[i])
        # update yj[i]
        for j in Iu:
            yj[j] += lr_yj * (loss * qi[i] / sqrt_iu - reg_yj * yj[j])
    return (('bi', bi), ('bu', bu), ('qi', qi), ('pu', pu), ('yj', yj))


def map_loss(
    indices: list[tuple[int, int]],
    ratings: list[float | int],
    global_mean: int | float,
    bi: np.ndarray,
    bu: np.ndarray,
    qi: np.ndarray,
    pu: np.ndarray,
    u_impl_fdb: np.ndarray,
) -> list[tuple[int, int, float]]:
    """Compute loss."""
    # check parameters
    if not isinstance(indices, list):
        raise TypeError
    if not all(isinstance(i, int) and isinstance(u, int) and i >= 0 and u >= 0 for (i, u) in indices):
        raise ValueError
    if not all(isinstance(r, (float, int)) for r in ratings):
        raise TypeError
    if not isinstance(global_mean, (int, float)):
        raise TypeError
    if not isinstance(bi, np.ndarray):
        raise TypeError
    if not isinstance(bu, np.ndarray):
        raise TypeError
    if not isinstance(qi, np.ndarray):
        raise TypeError
    if not isinstance(pu, np.ndarray):
        raise TypeError
    if not isinstance(u_impl_fdb, np.ndarray):
        raise TypeError

    loss_list = []
    for (i, u), r in zip(indices, ratings):
        # compute loss
        dot = 0.0
        dot += qi[i].dot(pu[u] + u_impl_fdb[u])
        loss = r - (global_mean + 1.5 * bu[u] + bi[i] + dot)
        loss_list.append((i, u, loss))
    return loss_list


def get_top_n_items(
    indices: tuple[tuple[int, int], tuple[int, int]],
    top_n: int,
    global_mean: int | float,
    bi: np.ndarray,
    bu: np.ndarray,
    qi: np.ndarray,
    pu: np.ndarray,
    u_impl_fdb: np.ndarray,
    item_index_id: dict,
    user_index_id: dict,
) -> list[tuple[int, int, float, int]]:
    """Get top-N items for each user."""
    # check parameters
    if not isinstance(indices, tuple):
        raise TypeError
    if not isinstance(indices[0], tuple):
        raise TypeError
    if not isinstance(indices[1], tuple):
        raise TypeError
    if not isinstance(indices[0][0], int):
        raise TypeError
    if not isinstance(indices[0][1], int):
        raise TypeError
    if not isinstance(indices[1][0], int):
        raise TypeError
    if not isinstance(indices[1][1], int):
        raise TypeError
    if not isinstance(top_n, int) and top_n <= 0:
        raise ValueError
    if not isinstance(global_mean, (int, float)):
        raise TypeError
    if not isinstance(bi, np.ndarray):
        raise TypeError
    if not isinstance(bu, np.ndarray):
        raise TypeError
    if not isinstance(qi, np.ndarray):
        raise TypeError
    if not isinstance(pu, np.ndarray):
        raise TypeError
    if not isinstance(u_impl_fdb, np.ndarray):
        raise TypeError
    if not isinstance(item_index_id, dict):
        raise TypeError
    if not all((isinstance(index, int) and index >= 0) for index in item_index_id.keys()):
        raise ValueError
    if not isinstance(user_index_id, dict):
        raise TypeError
    if not all((isinstance(index, int) and index >= 0) for index in user_index_id.keys()):
        raise ValueError

    i_start, i_end = indices[0]
    u_start, u_end = indices[1]
    pu = pu[u_start:u_end]
    u_impl_fdb = u_impl_fdb[u_start:u_end]
    qi = qi[i_start:i_end]
    bu = bu[u_start:u_end]
    bi = bi[i_start:i_end]
    iumat = global_mean + (1.5 * bu) + bi.reshape(bi.size, 1) + qi.dot((pu + u_impl_fdb).T)
    uimat = iumat.T
    urmat = np.argsort(-uimat, axis=1)[:, :top_n]
    output = []
    for user_idx, item_indices in enumerate(urmat):
        user_id = user_index_id[user_idx + u_start]
        for rank, item_idx in enumerate(item_indices):
            rating_pred = float(uimat[user_idx, item_idx])
            item_id = item_index_id[item_idx + i_start]
            output.append((item_id, user_id, rating_pred, rank + 1))
    return output


def run(
    spark: pyspark.sql.SparkSession, n_pars: int, input_data_path: str, output_data_path: str | None
) -> None:
    """Run the application."""
    # check parameters
    if not isinstance(spark, pyspark.sql.SparkSession):
        raise TypeError
    if not isinstance(n_pars, int) and n_pars <= 0:
        raise ValueError
    if not isinstance(input_data_path, str):
        raise ValueError
    if not isinstance(output_data_path, str):
        raise ValueError
    if not input_data_path.endswith('.parquet'):
        raise ValueError
    if not output_data_path.endswith('.parquet'):
        raise ValueError

    # 1. Set up arguments
    # data arguments
    item_col = 'i'
    user_col = 'u'
    rating_col = 'r'
    # algorithm arguments
    n_samples_per_partition = None
    n_epochs = 1
    n_factors = 32
    top_n = 10
    rstate = np.random.RandomState(9999)
    lr_bu = 0.007
    lr_bi = 0.007
    lr_qi = 0.007
    lr_pu = 0.007
    lr_yj = 0.007
    reg_bu = 0.02
    reg_bi = 0.02
    reg_qi = 0.02
    reg_pu = 0.02
    reg_yj = 0.02

    # 2. Set up SparkSession
    sc = spark.sparkContext
    sc.setLogLevel(DEFAULT_LOGLEVEL)

    # 3. Load data and transform data into spark dataframe dtype
    print('load data')
    dataset = Dataset()
    dataset.load_dataset(spark, input_data_path, item_col, user_col, rating_col, n_pars)
    n_items = dataset.n_items
    n_users = dataset.n_users
    indents = ' ' * 4
    print(f'{indents}number of items in the training data: {n_items}')
    print(f'{indents}number of users in the training data: {n_users}')
    train = dataset.train  # a list of (item_idx, user_idx, rating) tuples
    print('cache the training data')
    train.cache().count()

    # 4. Initialize and broadcast parameters
    # 4.1 initialize parameters
    print('initialize the parameters')
    ur = dataset.ur
    global_mean = train.flatMap(lambda x: x[1]).mean()
    assert isinstance(global_mean, float)
    assert global_mean > 0
    bu = np.zeros(n_users, np.float32)
    bi = np.zeros(n_items, np.float32)
    qi = rstate.normal(0, 0.1, (n_items, n_factors)).astype(np.float32)
    pu = rstate.normal(0, 0.1, (n_users, n_factors)).astype(np.float32)
    yj = rstate.normal(0, 0.1, (n_items, n_factors)).astype(np.float32)

    print('compute the user_implicit_feedback')
    u_impl_fdb = get_u_impl_fdb(n_users, n_factors, ur, yj)
    assert isinstance(u_impl_fdb, np.ndarray)
    assert u_impl_fdb.shape == (n_users, n_factors)

    # 4.2 broadcast the parameters to the cluster
    print('broadcast the parameters to the cluster')
    globla_mean_bcast = sc.broadcast(global_mean)
    bu_bcast = sc.broadcast(bu)
    bi_bcast = sc.broadcast(bi)
    qi_bcast = sc.broadcast(qi)
    pu_bcast = sc.broadcast(pu)
    yj_bcast = sc.broadcast(yj)
    u_impl_fdb_bcast = sc.broadcast(u_impl_fdb)
    item_index_id_bcast = sc.broadcast(dataset.item_index_id)
    user_index_id_bcast = sc.broadcast(dataset.user_index_id)

    # 5. Update the parameters
    ur_bcast = sc.broadcast(ur)
    for current_epoch in range(n_epochs):
        print(f'epoch {current_epoch+1}')
        t1 = time.time()
        # 5.1 get a sampled subset of training data
        if n_samples_per_partition is not None:
            samples_rdd = dataset.get_samples(k=n_samples_per_partition)
        else:
            samples_rdd = train
        # 5.2 compute loss
        loss_rdd = samples_rdd.map(
            lambda x: map_loss(
                indices=x[0],
                ratings=x[1],
                u_impl_fdb=u_impl_fdb_bcast.value,
                global_mean=globla_mean_bcast.value,
                bi=bi_bcast.value,
                bu=bu_bcast.value,
                qi=qi_bcast.value,
                pu=pu_bcast.value,
            )
        )

        n_samples = loss_rdd.map(len).sum()
        print(f'{indents}number of samples: {n_samples}')

        # 5.3 update parameters
        params_rdd = loss_rdd.map(
            lambda loss_list: map_sgd(
                loss_list=loss_list,
                bi=bi_bcast.value,
                bu=bu_bcast.value,
                qi=qi_bcast.value,
                pu=pu_bcast.value,
                yj=yj_bcast.value,
                u_impl_fdb=u_impl_fdb_bcast.value,
                lr_bi=lr_bi,
                lr_bu=lr_bu,
                lr_qi=lr_qi,
                lr_pu=lr_pu,
                lr_yj=lr_yj,
                reg_bi=reg_bi,
                reg_bu=reg_bu,
                reg_qi=reg_qi,
                reg_pu=reg_pu,
                reg_yj=reg_yj,
                ur=ur_bcast.value,
            )
        )

        # 5.4 compute the average of the parameters
        print(f'{indents}compute the average of the parameters')
        n_paramgroups = params_rdd.getNumPartitions()
        params_dict = (
            params_rdd.flatMap(lambda x: x)
            .reduceByKey(lambda arr1, arr2: arr1 + arr2)
            .mapValues(lambda arr: arr / n_paramgroups)
            .collectAsMap()
        )
        bi = params_dict['bi']
        bu = params_dict['bu']
        qi = params_dict['qi']
        pu = params_dict['pu']
        yj = params_dict['yj']
        print(f'{indents}compute user_implicit_feedback')
        u_impl_fdb = get_u_impl_fdb(n_users, n_factors, ur, yj)

        print(f'{indents}{time.time() - t1}s')

        # 5.5 broadcast the parameters to the cluster
        print(f'{indents}broadcast the parameters to the cluster')
        bi_bcast = sc.broadcast(bi)
        bu_bcast = sc.broadcast(bu)
        qi_bcast = sc.broadcast(qi)
        pu_bcast = sc.broadcast(pu)
        yj_bcast = sc.broadcast(yj)
        u_impl_fdb_bcast = sc.broadcast(u_impl_fdb)

    # 6. compute top-N items for each user
    print('compute top-N items for each user')
    rec_params = {
        'top_n': top_n,
        'global_mean': globla_mean_bcast.value,
        'bu': bu_bcast.value,
        'bi': bi_bcast.value,
        'qi': qi_bcast.value,
        'pu': pu_bcast.value,
        'item_index_id': item_index_id_bcast.value,
        'user_index_id': user_index_id_bcast.value,
        'u_impl_fdb': u_impl_fdb_bcast.value,
    }
    # 6.1 create the block information
    u_batch = 500
    block_info = []
    for x in range(0, n_users // u_batch + 1):
        i_indices = (0, n_items)
        u_indices = (x * u_batch, min(n_users, (x + 1) * u_batch))
        block_info.append((i_indices, u_indices))
    output_rdd = (
        sc.parallelize(block_info, numSlices=min(len(block_info), n_pars))
        .map(lambda indices: get_top_n_items(indices=indices, **rec_params))
        .flatMap(lambda x: x)
    )

    # 7. create a spark dataframe
    schema = StructType(
        [
            StructField(item_col, StringType()),
            StructField(user_col, StringType()),
            StructField(str(rating_col) + '_pred', DoubleType()),
            StructField('rank', IntegerType()),
        ]
    )
    output_sdf = output_rdd.toDF(schema)
    output_n_rows = output_sdf.count()
    # output_n_items = output_sdf.select([item_col]).distinct().count()
    # output_n_users = output_sdf.select([user_col]).distinct().count()
    print(f'number of rows: {output_n_rows}')
    # print(f'number of items: {output_n_items}')
    # print(f'number of users: {output_n_users}')
    output_sdf.show(n=20)

    # 8. write the result to the HDFS
    if output_data_path is not None:
        pass
        # cdt = datetime.today().strftime('%Y%m%d%H%M%S')
        # new_output_path = (
        #     '.'.join(output_data_path.split('.')[:-1] + [cdt, 'parquet'])
        # )
        # print(f'write the result to {new_output_path}')
        # output_sdf.write.parquet(new_output_path, mode='error')
