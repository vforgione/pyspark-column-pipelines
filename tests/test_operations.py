from datetime import datetime
from time import mktime

from pyspark.sql import SparkSession

from pyspark_column_pipelines import ColumnPipeline
from pyspark_column_pipelines.operations import Cast, FromUnixtime


def test__cast(spark: SparkSession) -> None:
    """Asserts the custom cast operation casts the column type"""
    u = datetime.utcnow()  # pylint: disable=invalid-name
    h = datetime.now()  # pylint: disable=invalid-name

    utc_now = datetime(u.year, u.month, u.day, u.hour, u.minute, u.second)
    here_now = datetime(h.year, h.month, h.day, h.hour, h.minute, h.second)

    srcdf = spark.createDataFrame(
        [
            (mktime(utc_now.timetuple()), mktime(here_now.timetuple())),
        ],
        ["utc", "local"],
    )

    pipeline = ColumnPipeline(
        FromUnixtime(),
        Cast("timestamp"),
    )
    result = pipeline.execute(srcdf, "utc")
    result = pipeline.execute(result, "local")

    expected = spark.createDataFrame(
        [
            (utc_now, here_now),
        ],
        ["utc", "local"],
    )

    assert set(result.collect()) == set(expected.collect())
