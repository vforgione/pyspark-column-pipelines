from pyspark.sql import DataFrame

from .operations import ColumnPipelineOperation
from .pipelines import ColumnPipeline

# monkey patch the DataFrame class


def with_column_pipeline(
    self: DataFrame, col: str, *ops: ColumnPipelineOperation
) -> DataFrame:
    """Provides an inline means of creating a column pipeline

    :param col: The name of the column to reauthor
    :param ops: A series of transformation operations
    :returns: The dataframe with transformations scheduled for the column
    """
    pipeline = ColumnPipeline(*ops)
    return pipeline.execute(self, col)


DataFrame.withColumnPipeline = with_column_pipeline  # type: ignore
DataFrame.with_column_pipeline = with_column_pipeline  # type: ignore
