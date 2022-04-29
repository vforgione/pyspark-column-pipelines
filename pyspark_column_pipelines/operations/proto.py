from typing import Protocol

from pyspark.sql import Column

__all__ = ["ColumnPipelineOperation"]


class ColumnPipelineOperation(Protocol):
    """Provides an API for transformation operations within a `ColumnPipeline`"""

    def execute(self, col: str) -> Column:
        """Executes a configured `pyspark.sql.function` transformation using the given
        column name and any additional class-specific attributes as arguments to the
        transformation function.

        :param col: The target column name
        :returns: The output of the spark function
        """
