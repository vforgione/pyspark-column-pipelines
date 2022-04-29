from functools import reduce

from pyspark.sql import DataFrame

from .operations.proto import ColumnPipelineOperation

__all__ = ["ColumnPipeline"]


class ColumnPipeline:
    """Execute a series of transformations for a specified column

    :attr ops: The transformation functions that operate on the column
    """

    def __init__(self, *ops: ColumnPipelineOperation):
        self.ops = ops

    def execute(self, df: DataFrame, col: str) -> DataFrame:
        """Iterates the pipeline operations and executes them

        :param df: The dataframe to use
        :param col: The name of the column to reauthor
        :returns: The dataframe with transformations scheduled for the column
        """
        return reduce(lambda d, op: d.withColumn(col, op.execute(col)), self.ops, df)
