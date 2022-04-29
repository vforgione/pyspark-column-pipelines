from pyspark.sql import Column
from pyspark.sql import functions as F

from .proto import ColumnPipelineOperation


class Cast(ColumnPipelineOperation):
    """Casts a column to a given data type

    :attr data_type: The target data type
    """

    def __init__(self, data_type: str):
        self.data_type = data_type

    def execute(self, col: str) -> Column:
        return F.col(col).cast(self.data_type)
