from pyspark.sql import SparkSession

from pyspark_column_pipelines import ColumnPipeline
from pyspark_column_pipelines.operations import Cast


def test__execute__returns_df(spark: SparkSession) -> None:
    """Asserts the dataframe returned is a transformation of the original"""
    srcdf = spark.createDataFrame(
        [
            (1, "one", "uno"),
            (2, "two", "dos"),
            (3, "three", "tres"),
        ],
        ["numeral", "en", "es"],
    )
    pipeline = ColumnPipeline(Cast("string"))
    result = pipeline.execute(srcdf, "numeral")

    expected = spark.createDataFrame(
        [
            ("1", "one", "uno"),
            ("2", "two", "dos"),
            ("3", "three", "tres"),
        ],
        ["numeral", "en", "es"],
    )

    assert set(result.collect()) == set(expected.collect())


def test__monkey_patch(spark: SparkSession) -> None:
    """Asserts that the monkey patch applied in the package init works"""
    srcdf = spark.createDataFrame(
        [
            (1, "one", "uno"),
            (2, "two", "dos"),
            (3, "three", "tres"),
        ],
        ["numeral", "en", "es"],
    )
    result = srcdf.withColumnPipeline("numeral", Cast("string"))  # type: ignore

    expected = spark.createDataFrame(
        [
            ("1", "one", "uno"),
            ("2", "two", "dos"),
            ("3", "three", "tres"),
        ],
        ["numeral", "en", "es"],
    )

    assert set(result.collect()) == set(expected.collect())
