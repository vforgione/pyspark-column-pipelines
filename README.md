# PySpark Column Pipelines

PySpark Column Pipelines provides data engineers a simple way to define a series
of transformations and apply it as needed to as many columns needed.

```python
from pyspark_column_pipeline import ColumnPipeline
from pyspark_column_pipeline.operations import Cast, FromUnixtime

pipeline = ColumnPipeline(
    FromUnixtime(),
    Cast("timestamp"),
)

df = pipeline.execute(df, "utc_timestamp")
df = pipeline.execute(df, "local_timestamp")
```

## Package API

This package provides the pipeline class, a custom _Cast_ operation, and wrapper
operations for all
[pyspark.sql.function](https://github.com/apache/spark/blob/master/python/pyspark/sql/functions.py)
transformations.

Pipelines are simple objects: they hold a sequence of operations and then execute those
operations against a target dataframe and column using
[functools reduce](https://docs.python.org/3/library/functools.html#functools.reduce).

Operations are simple wrappers for any sort of transformation operation that can be
applied to a dataset column. This can be easily extended to use any custom action or
UDF desired (as demonstrated with the _Cast_ operation).

Also worth noting is a monkey patch to the `DataFrame` class. Pipelines can be written
inline "natively" on a dataframe:

```python
df = df.withColumnPipeline("some_col", Op1(), Op2(), Op3(), ...)
```

## Disclosures

This was created to scratch an itch. It works for what I intend it to do, and I'm not
absolutely sure I want to fully support this. I will not contribute this software as it
currently exists to PyPI, but if you want to source it directly from this repository
and use it go ahead.
