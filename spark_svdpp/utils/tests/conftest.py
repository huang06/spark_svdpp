import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope='session')
def spark(request):
    """Fxiture for creating a spark session."""
    spark = (
        SparkSession.builder
        .master('local[2]')
        .appName('pytest-pyspark-local')
        .getOrCreate()
    )
    request.addfinalizer(lambda: spark.sparkContext.stop())
    return spark
