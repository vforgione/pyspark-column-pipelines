import pytest
from pyspark.sql import SparkSession


@pytest.fixture
def spark() -> SparkSession:
    """Creates a spark session for tests"""
    return SparkSession.builder.appName("tests").master("local[*]").getOrCreate()
