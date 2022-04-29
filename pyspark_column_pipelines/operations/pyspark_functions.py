from pyspark.sql import Column
from pyspark.sql import functions as F

from .proto import ColumnPipelineOperation

__all__ = [
    "Abs",
    "Acos",
    "Acosh",
    "AddMonths",
    "Aggregate",
    "ApproxCountDistinct",
    "Array",
    "ArrayContains",
    "ArrayDistinct",
    "ArrayExcept",
    "ArrayIntersect",
    "ArrayJoin",
    "ArrayMax",
    "ArrayMin",
    "ArrayPosition",
    "ArrayRemove",
    "ArrayRepeat",
    "ArraySort",
    "ArrayUnion",
    "ArraysOverlap",
    "ArraysZip",
    "Asc",
    "AscNullsFirst",
    "AscNullsLast",
    "Ascii",
    "Asin",
    "Asinh",
    "AssertTrue",
    "Atan",
    "Atan2",
    "Atanh",
    "Avg",
    "Base64",
    "Bin",
    "BitwiseNot",
    "Bround",
    "Cbrt",
    "Ceil",
    "Coalesce",
    "Col",
    "CollectList",
    "CollectSet",
    "Concat",
    "ConcatWs",
    "Conv",
    "Corr",
    "Cos",
    "Cosh",
    "Count",
    "CountDistinct",
    "CovarPop",
    "CovarSamp",
    "Crc32",
    "CreateMap",
    "CumeDist",
    "CurrentDate",
    "CurrentTimestamp",
    "DateAdd",
    "DateFormat",
    "DateSub",
    "DateTrunc",
    "Datediff",
    "DayOfMonth",
    "DayOfWeek",
    "DayOfYear",
    "Decode",
    "Degrees",
    "DenseRank",
    "Desc",
    "DescNullsFirst",
    "DescNullsLast",
    "ElementAt",
    "Encode",
    "Exists",
    "Exp",
    "Explode",
    "ExplodeOuter",
    "Expm1",
    "Expr",
    "Factorial",
    "Filter",
    "First",
    "Flatten",
    "Floor",
    "Forall",
    "FormatNumber",
    "FormatString",
    "FromCsv",
    "FromJson",
    "FromUnixtime",
    "FromUtcTimestamp",
    "GetJsonObject",
    "Greatest",
    "Grouping",
    "GroupingId",
    "Hash",
    "Hex",
    "Hour",
    "Hypot",
    "Initcap",
    "InputFileName",
    "Instr",
    "IsNan",
    "IsNull",
    "JsonTuple",
    "Kurtosis",
    "Lag",
    "Last",
    "LastDay",
    "Lead",
    "Least",
    "Length",
    "Levenshtein",
    "Lit",
    "Locate",
    "Log",
    "Log10",
    "Log1P",
    "Log2",
    "Lower",
    "Lpad",
    "Ltrim",
    "MapConcat",
    "MapEntries",
    "MapFilter",
    "MapFromArrays",
    "MapFromEntries",
    "MapKeys",
    "MapValues",
    "MapZipWith",
    "Max",
    "Md5",
    "Mean",
    "Min",
    "Minute",
    "MonotonicallyIncreasingId",
    "Month",
    "MonthsBetween",
    "Nanvl",
    "NextDay",
    "NthValue",
    "Ntile",
    "Overlay",
    "PercentRank",
    "PercentileApprox",
    "Posexplode",
    "PosexplodeOuter",
    "Pow",
    "Product",
    "Quarter",
    "Radians",
    "RaiseError",
    "Rand",
    "Randn",
    "Rank",
    "RegexpExtract",
    "RegexpReplace",
    "Repeat",
    "Reverse",
    "Rint",
    "Round",
    "RowNumber",
    "Rpad",
    "Rtrim",
    "SchemaOfCsv",
    "SchemaOfJson",
    "Second",
    "Sentences",
    "Sequence",
    "SessionWindow",
    "Sha1",
    "Sha2",
    "Shiftleft",
    "Shiftright",
    "Shiftrightunsigned",
    "Shuffle",
    "Signum",
    "Sin",
    "Sinh",
    "Size",
    "Skewness",
    "Slice",
    "SortArray",
    "Soundex",
    "SparkPartitionId",
    "Split",
    "Sqrt",
    "Stddev",
    "StddevPop",
    "StddevSamp",
    "Struct",
    "Substring",
    "SubstringIndex",
    "Sum",
    "SumDistinct",
    "Tan",
    "Tanh",
    "TimestampSeconds",
    "ToCsv",
    "ToDate",
    "ToJson",
    "ToTimestamp",
    "ToUtcTimestamp",
    "Transform",
    "TransformKeys",
    "TransformValues",
    "Translate",
    "Trim",
    "Trunc",
    "Unbase64",
    "Unhex",
    "UnixTimestamp",
    "Upper",
    "VarPop",
    "VarSamp",
    "Variance",
    "WeekOfYear",
    "When",
    "Window",
    "Xxhash64",
    "Year",
    "ZipWith",
]


class Abs(ColumnPipelineOperation):
    """Computes the absolute value."""

    def execute(self, col: str) -> Column:
        return F.abs(col)


class Acos(ColumnPipelineOperation):
    """inverse cosine of `col`, as if computed by `java.lang.Math.acos()`"""

    def execute(self, col: str) -> Column:
        return F.acos(col)


class Acosh(ColumnPipelineOperation):
    """Computes inverse hyperbolic cosine of the input column."""

    def execute(self, col: str) -> Column:
        return F.acosh(col)


class AddMonths(ColumnPipelineOperation):
    """Returns the date that is `months` months after `start`"""

    def __init__(self, months):
        self.months = months

    def execute(self, col: str) -> Column:
        return F.add_months(col, self.months)


class Aggregate(ColumnPipelineOperation):
    """Applies a binary operator to an initial state and all elements in the array, and
    reduces this to a single state. The final state is converted into the final
    result by applying a finish function."""

    def __init__(self, initial_value, merge, finish=None):
        self.initial_value = initial_value
        self.merge = merge
        self.finish = finish

    def execute(self, col: str) -> Column:
        return F.aggregate(col, self.initial_value, self.merge, self.finish)


class ApproxCountDistinct(ColumnPipelineOperation):
    """Aggregate function: returns a new :class:`~pyspark.sql.Column` for approximate
    distinct count of column `col`."""

    def __init__(self, rsd=None):
        self.rsd = rsd

    def execute(self, col: str) -> Column:
        return F.approx_count_distinct(col, self.rsd)


class Array(ColumnPipelineOperation):
    """Creates a new array column."""

    def __init__(self, *cols):
        self.cols = cols

    def execute(self, _) -> Column:
        return F.array(*self.cols)


class ArrayContains(ColumnPipelineOperation):
    """Collection function: returns null if the array is null, true if the array
    contains the given value, and false otherwise."""

    def __init__(self, value):
        self.value = value

    def execute(self, col: str) -> Column:
        return F.array_contains(col, self.value)


class ArrayDistinct(ColumnPipelineOperation):
    """Collection function: removes duplicate values from the array."""

    def execute(self, col: str) -> Column:
        return F.array_distinct(col)


class ArrayExcept(ColumnPipelineOperation):
    """Collection function: returns an array of the elements in col1 but not in col2,
    without duplicates."""

    def __init__(self, col2):
        self.col2 = col2

    def execute(self, col: str) -> Column:
        return F.array_except(col, self.col2)


class ArrayIntersect(ColumnPipelineOperation):
    """Collection function: returns an array of the elements in the intersection of
    col1 and col2, without duplicates."""

    def __init__(self, col2):
        self.col2 = col2

    def execute(self, col: str) -> Column:
        return F.array_intersect(col, self.col2)


class ArrayJoin(ColumnPipelineOperation):
    """Concatenates the elements of `column` using the `delimiter`. Null values are
    replaced with `null_replacement` if set, otherwise they are ignored."""

    def __init__(self, delimiter, null_replacement=None):
        self.delimiter = delimiter
        self.null_replacement = null_replacement

    def execute(self, col: str) -> Column:
        return F.array_join(col, self.delimiter, self.null_replacement)


class ArrayMax(ColumnPipelineOperation):
    """Collection function: returns the maximum value of the array."""

    def execute(self, col: str) -> Column:
        return F.array_max(col)


class ArrayMin(ColumnPipelineOperation):
    """Collection function: returns the minimum value of the array."""

    def execute(self, col: str) -> Column:
        return F.array_min(col)


class ArrayPosition(ColumnPipelineOperation):
    """Collection function: Locates the position of the first occurrence of the given
    value in the given array. Returns null if either of the arguments are null."""

    def __init__(self, value):
        self.value = value

    def execute(self, col: str) -> Column:
        return F.array_position(col, self.value)


class ArrayRemove(ColumnPipelineOperation):
    """Collection function: Remove all elements that equal to element from the given
    array."""

    def __init__(self, element):
        self.element = element

    def execute(self, col: str) -> Column:
        return F.array_remove(col, self.element)


class ArrayRepeat(ColumnPipelineOperation):
    """Collection function: creates an array containing a column repeated count times."""

    def __init__(self, count):
        self.count = count

    def execute(self, col: str) -> Column:
        return F.array_repeat(col, self.count)


class ArraySort(ColumnPipelineOperation):
    """Collection function: sorts the input array in ascending order. The elements of
    the input array must be orderable. Null elements will be placed at the end
    of the returned array."""

    def execute(self, col: str) -> Column:
        return F.array_sort(col)


class ArrayUnion(ColumnPipelineOperation):
    """Collection function: returns an array of the elements in the union of col1 and
    col2, without duplicates."""

    def __init__(self, col2):
        self.col2 = col2

    def execute(self, col: str) -> Column:
        return F.array_union(col, self.col2)


class ArraysOverlap(ColumnPipelineOperation):
    """Collection function: returns true if the arrays contain any common non-null
    element; if not, returns null if both the arrays are non-empty and any of
    them contains a null element; returns false otherwise."""

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def execute(self, _) -> Column:
        return F.arrays_overlap(self.left, self.right)


class ArraysZip(ColumnPipelineOperation):
    """Collection function: Returns a merged array of structs in which the N-th struct
    contains all N-th values of input arrays."""

    def __init__(self, *cols):
        self.cols = cols

    def execute(self, _) -> Column:
        return F.arrays_zip(*self.cols)


class Asc(ColumnPipelineOperation):
    """Returns a sort expression based on the ascending order of the given column name."""

    def execute(self, col: str) -> Column:
        return F.asc(col)


class AscNullsFirst(ColumnPipelineOperation):
    """Returns a sort expression based on the ascending order of the given column name,
    and null values return before non-null values."""

    def execute(self, col: str) -> Column:
        return F.asc_nulls_first(col)


class AscNullsLast(ColumnPipelineOperation):
    """Returns a sort expression based on the ascending order of the given column name,
    and null values appear after non-null values."""

    def execute(self, col: str) -> Column:
        return F.asc_nulls_last(col)


class Ascii(ColumnPipelineOperation):
    """Computes the numeric value of the first character of the string column."""

    def execute(self, col: str) -> Column:
        return F.ascii(col)


class Asin(ColumnPipelineOperation):
    """inverse sine of `col`, as if computed by `java.lang.Math.asin()`"""

    def execute(self, col: str) -> Column:
        return F.asin(col)


class Asinh(ColumnPipelineOperation):
    """Computes inverse hyperbolic sine of the input column."""

    def execute(self, col: str) -> Column:
        return F.asinh(col)


class AssertTrue(ColumnPipelineOperation):
    """Returns null if the input column is true; throws an exception with the provided
    error message otherwise."""

    def __init__(self, err_msg=None):
        self.err_msg = err_msg

    def execute(self, col: str) -> Column:
        return F.assert_true(col, self.err_msg)


class Atan(ColumnPipelineOperation):
    """inverse tangent of `col`, as if computed by `java.lang.Math.atan()`"""

    def execute(self, col: str) -> Column:
        return F.atan(col)


class Atan2(ColumnPipelineOperation):
    """the `theta` component of the point (`r`, `theta`) in polar coordinates that
    corresponds to the point (`x`, `y`) in Cartesian coordinates, as if computed
    by `java.lang.Math.atan2()`"""

    def __init__(self, col2):
        self.col2 = col2

    def execute(self, col: str) -> Column:
        return F.atan2(col, self.col2)


class Atanh(ColumnPipelineOperation):
    """Computes inverse hyperbolic tangent of the input column."""

    def execute(self, col: str) -> Column:
        return F.atanh(col)


class Avg(ColumnPipelineOperation):
    """Aggregate function: returns the average of the values in a group."""

    def execute(self, col: str) -> Column:
        return F.avg(col)


class Base64(ColumnPipelineOperation):
    """Computes the BASE64 encoding of a binary column and returns it as a string
    column."""

    def execute(self, col: str) -> Column:
        return F.base64(col)


class Bin(ColumnPipelineOperation):
    """Returns the string representation of the binary value of the given column."""

    def execute(self, col: str) -> Column:
        return F.bin(col)


class BitwiseNot(ColumnPipelineOperation):
    """Computes bitwise not."""

    def execute(self, col: str) -> Column:
        return F.bitwise_not(col)


class Bround(ColumnPipelineOperation):
    """Round the given value to `scale` decimal places using HALF_EVEN rounding mode if
    `scale` >= 0 or at integral part when `scale` < 0."""

    def __init__(self, scale=0):
        self.scale = scale

    def execute(self, col: str) -> Column:
        return F.bround(col, self.scale)


class Cbrt(ColumnPipelineOperation):
    """Computes the cube-root of the given value."""

    def execute(self, col: str) -> Column:
        return F.cbrt(col)


class Ceil(ColumnPipelineOperation):
    """Computes the ceiling of the given value."""

    def execute(self, col: str) -> Column:
        return F.ceil(col)


class Coalesce(ColumnPipelineOperation):
    """Returns the first column that is not null."""

    def __init__(self, *cols):
        self.cols = cols

    def execute(self, _) -> Column:
        return F.coalesce(*self.cols)


class Col(ColumnPipelineOperation):
    """Returns a :class:`~pyspark.sql.Column` based on the given column name.' Examples
    -------- >>> col('x') Column<'x'> >>> column('x') Column<'x'>"""

    def __init__(self, col):
        self.col = col

    def execute(self, _) -> Column:
        return F.col(self.col)


class CollectList(ColumnPipelineOperation):
    """Aggregate function: returns a list of objects with duplicates."""

    def execute(self, col: str) -> Column:
        return F.collect_list(col)


class CollectSet(ColumnPipelineOperation):
    """Aggregate function: returns a set of objects with duplicate elements eliminated."""

    def execute(self, col: str) -> Column:
        return F.collect_set(col)


class Concat(ColumnPipelineOperation):
    """Concatenates multiple input columns together into a single column. The function
    works with strings, binary and compatible array columns."""

    def __init__(self, *cols):
        self.cols = cols

    def execute(self, _) -> Column:
        return F.concat(*self.cols)


class ConcatWs(ColumnPipelineOperation):
    """Concatenates multiple input string columns together into a single string column,
    using the given separator."""

    def __init__(self, sep, *cols):
        self.sep = sep
        self.cols = cols

    def execute(self, _) -> Column:
        return F.concat_ws(self.sep, *self.cols)


class Conv(ColumnPipelineOperation):
    """Convert a number in a string column from one base to another."""

    def __init__(self, from_base, to_base):
        self.from_base = from_base
        self.to_base = to_base

    def execute(self, col: str) -> Column:
        return F.conv(col, self.from_base, self.to_base)


class Corr(ColumnPipelineOperation):
    """Returns a new :class:`~pyspark.sql.Column` for the Pearson Correlation
    Coefficient for ``col1`` and ``col2``."""

    def __init__(self, col2):
        self.col2 = col2

    def execute(self, col: str) -> Column:
        return F.corr(col, self.col2)


class Cos(ColumnPipelineOperation):
    """cosine of the angle, as if computed by `java.lang.Math.cos()`."""

    def execute(self, col: str) -> Column:
        return F.cos(col)


class Cosh(ColumnPipelineOperation):
    """hyperbolic cosine of the angle, as if computed by `java.lang.Math.cosh()`"""

    def execute(self, col: str) -> Column:
        return F.cosh(col)


class Count(ColumnPipelineOperation):
    """Aggregate function: returns the number of items in a group."""

    def execute(self, col: str) -> Column:
        return F.count(col)


class CountDistinct(ColumnPipelineOperation):
    """Returns a new :class:`Column` for distinct count of ``col`` or ``cols``."""

    def __init__(self, *cols):
        self.cols = cols

    def execute(self, col: str) -> Column:
        return F.count_distinct(col, *self.cols)


class CovarPop(ColumnPipelineOperation):
    """Returns a new :class:`~pyspark.sql.Column` for the population covariance of
    ``col1`` and ``col2``."""

    def __init__(self, col2):
        self.col2 = col2

    def execute(self, col: str) -> Column:
        return F.covar_pop(col, self.col2)


class CovarSamp(ColumnPipelineOperation):
    """Returns a new :class:`~pyspark.sql.Column` for the sample covariance of ``col1``
    and ``col2``."""

    def __init__(self, col2):
        self.col2 = col2

    def execute(self, col: str) -> Column:
        return F.covar_samp(col, self.col2)


class Crc32(ColumnPipelineOperation):
    """Calculates the cyclic redundancy check value  (CRC32) of a binary column and
    returns the value as a bigint."""

    def execute(self, col: str) -> Column:
        return F.crc32(col)


class CreateMap(ColumnPipelineOperation):
    """Creates a new map column."""

    def __init__(self, *cols):
        self.cols = cols

    def execute(self, _) -> Column:
        return F.create_map(*self.cols)


class CumeDist(ColumnPipelineOperation):
    """Window function: returns the cumulative distribution of values within a window
    partition, i.e. the fraction of rows that are below the current row."""

    def execute(self, _) -> Column:
        return F.cume_dist()


class CurrentDate(ColumnPipelineOperation):
    """Returns the current date at the start of query evaluation as a :class:`DateType`
    column. All calls of current_date within the same query return the same
    value."""

    def execute(self, _) -> Column:
        return F.current_date()


class CurrentTimestamp(ColumnPipelineOperation):
    """Returns the current timestamp at the start of query evaluation as a
    :class:`TimestampType` column. All calls of current_timestamp within the
    same query return the same value."""

    def execute(self, _) -> Column:
        return F.current_timestamp()


class DateAdd(ColumnPipelineOperation):
    """Returns the date that is `days` days after `start`"""

    def __init__(self, days):
        self.days = days

    def execute(self, col: str) -> Column:
        return F.date_add(col, self.days)


class DateFormat(ColumnPipelineOperation):
    """Converts a date/timestamp/string to a value of string in the format specified by
    the date format given by the second argument."""

    def __init__(self, fmt):
        self.fmt = fmt

    def execute(self, col: str) -> Column:
        return F.date_format(col, self.fmt)


class DateSub(ColumnPipelineOperation):
    """Returns the date that is `days` days before `start`"""

    def __init__(self, days):
        self.days = days

    def execute(self, col: str) -> Column:
        return F.date_sub(col, self.days)


class DateTrunc(ColumnPipelineOperation):
    """Returns timestamp truncated to the unit specified by the format."""

    def __init__(self, fmt, timestamp):
        self.fmt = fmt
        self.timestamp = timestamp

    def execute(self, _) -> Column:
        return F.date_trunc(self.fmt, self.timestamp)


class Datediff(ColumnPipelineOperation):
    """Returns the number of days from `start` to `end`."""

    def __init__(self, end, start):
        self.end = end
        self.start = start

    def execute(self, _) -> Column:
        return F.datediff(self.end, self.start)


class DayOfMonth(ColumnPipelineOperation):
    """Extract the day of the month of a given date as integer."""

    def execute(self, col: str) -> Column:
        return F.dayofmonth(col)


class DayOfWeek(ColumnPipelineOperation):
    """Extract the day of the week of a given date as integer."""

    def execute(self, col: str) -> Column:
        return F.dayofweek(col)


class DayOfYear(ColumnPipelineOperation):
    """Extract the day of the year of a given date as integer."""

    def execute(self, col: str) -> Column:
        return F.dayofyear(col)


class Decode(ColumnPipelineOperation):
    """Computes the first argument into a string from a binary using the provided
    character set (one of 'US-ASCII', 'ISO-8859-1', 'UTF-8', 'UTF-16BE',
    'UTF-16LE', 'UTF-16')."""

    def __init__(self, charset):
        self.charset = charset

    def execute(self, col: str) -> Column:
        return F.decode(col, self.charset)


class Degrees(ColumnPipelineOperation):
    """Converts an angle measured in radians to an approximately equivalent angle
    measured in degrees."""

    def execute(self, col: str) -> Column:
        return F.degrees(col)


class DenseRank(ColumnPipelineOperation):
    """Window function: returns the rank of rows within a window partition, without any
    gaps."""

    def execute(self, _) -> Column:
        return F.dense_rank()


class Desc(ColumnPipelineOperation):
    """Returns a sort expression based on the descending order of the given column
    name."""

    def execute(self, col: str) -> Column:
        return F.desc(col)


class DescNullsFirst(ColumnPipelineOperation):
    """Returns a sort expression based on the descending order of the given column
    name, and null values appear before non-null values."""

    def execute(self, col: str) -> Column:
        return F.desc_nulls_first(col)


class DescNullsLast(ColumnPipelineOperation):
    """Returns a sort expression based on the descending order of the given column
    name, and null values appear after non-null values."""

    def execute(self, col: str) -> Column:
        return F.desc_nulls_last(col)


class ElementAt(ColumnPipelineOperation):
    """Collection function: Returns element of array at given index in extraction if
    col is array. Returns value for the given key in extraction if col is map."""

    def __init__(self, extraction):
        self.extraction = extraction

    def execute(self, col: str) -> Column:
        return F.element_at(col, self.extraction)


class Encode(ColumnPipelineOperation):
    """Computes the first argument into a binary from a string using the provided
    character set (one of 'US-ASCII', 'ISO-8859-1', 'UTF-8', 'UTF-16BE',
    'UTF-16LE', 'UTF-16')."""

    def __init__(self, charset):
        self.charset = charset

    def execute(self, col: str) -> Column:
        return F.encode(col, self.charset)


class Exists(ColumnPipelineOperation):
    """Returns whether a predicate holds for one or more elements in the array."""

    def __init__(self, func):
        self.func = func

    def execute(self, col: str) -> Column:
        return F.exists(col, self.func)


class Exp(ColumnPipelineOperation):
    """Computes the exponential of the given value."""

    def execute(self, col: str) -> Column:
        return F.exp(col)


class Explode(ColumnPipelineOperation):
    """Returns a new row for each element in the given array or map. Uses the default
    column name `col` for elements in the array and `key` and `value` for
    elements in the map unless specified otherwise."""

    def execute(self, col: str) -> Column:
        return F.explode(col)


class ExplodeOuter(ColumnPipelineOperation):
    """Returns a new row for each element in the given array or map. Unlike explode, if
    the array/map is null or empty then null is produced. Uses the default
    column name `col` for elements in the array and `key` and `value` for
    elements in the map unless specified otherwise."""

    def execute(self, col: str) -> Column:
        return F.explode_outer(col)


class Expm1(ColumnPipelineOperation):
    """Computes the exponential of the given value minus one."""

    def execute(self, col: str) -> Column:
        return F.expm1(col)


class Expr(ColumnPipelineOperation):
    """Parses the expression string into the column that it represents"""

    def __init__(self, string):
        self.string = string

    def execute(self, _) -> Column:
        return F.expr(self.string)


class Factorial(ColumnPipelineOperation):
    """Computes the factorial of the given value."""

    def execute(self, col: str) -> Column:
        return F.factorial(col)


class Filter(ColumnPipelineOperation):
    """Returns an array of elements for which a predicate holds in a given array."""

    def __init__(self, func):
        self.func = func

    def execute(self, col: str) -> Column:
        return F.filter(col, self.func)


class First(ColumnPipelineOperation):
    """Aggregate function: returns the first value in a group."""

    def __init__(self, ignorenulls=False):
        self.ignorenulls = ignorenulls

    def execute(self, col: str) -> Column:
        return F.first(col, self.ignorenulls)


class Flatten(ColumnPipelineOperation):
    """Collection function: creates a single array from an array of arrays. If a
    structure of nested arrays is deeper than two levels, only one level of
    nesting is removed."""

    def execute(self, col: str) -> Column:
        return F.flatten(col)


class Floor(ColumnPipelineOperation):
    """Computes the floor of the given value."""

    def execute(self, col: str) -> Column:
        return F.floor(col)


class Forall(ColumnPipelineOperation):
    """Returns whether a predicate holds for every element in the array."""

    def __init__(self, func):
        self.func = func

    def execute(self, col: str) -> Column:
        return F.forall(col, self.func)


class FormatNumber(ColumnPipelineOperation):
    """Formats the number X to a format like '#,--#,--#.--', rounded to d decimal
    places with HALF_EVEN round mode, and returns the result as a string."""

    def __init__(self, decimal_places):
        self.decimal_places = decimal_places

    def execute(self, col: str) -> Column:
        return F.format_number(col, self.decimal_places)


class FormatString(ColumnPipelineOperation):
    """Formats the arguments in printf-style and returns the result as a string column."""

    def __init__(self, fmt, *cols):
        self.fmt = fmt
        self.cols = cols

    def execute(self, _) -> Column:
        return F.format_string(self.fmt, *self.cols)


class FromCsv(ColumnPipelineOperation):
    """Parses a column containing a CSV string to a row with the specified schema.
    Returns `null`, in the case of an unparseable string."""

    def __init__(self, schema, options=None):
        self.schema = schema
        self.options = options

    def execute(self, col: str) -> Column:
        return F.from_csv(col, self.schema, self.options)


class FromJson(ColumnPipelineOperation):
    """Parses a column containing a JSON string into a :class:`MapType` with
    :class:`StringType` as keys type, :class:`StructType` or :class:`ArrayType`
    with the specified schema. Returns `null`, in the case of an unparseable
    string."""

    def __init__(self, schema, options=None):
        self.schema = schema
        self.options = options

    def execute(self, col: str) -> Column:
        return F.from_json(col, self.schema, self.options)


class FromUnixtime(ColumnPipelineOperation):
    """Converts the number of seconds from unix epoch (1970-01-01 00:00:00 UTC) to a
    string representing the timestamp of that moment in the current system time
    zone in the given format."""

    def __init__(self, fmt="yyyy-MM-dd HH:mm:ss"):
        self.fmt = fmt

    def execute(self, col: str) -> Column:
        return F.from_unixtime(col, self.fmt)


class FromUtcTimestamp(ColumnPipelineOperation):
    """This is a common function for databases supporting TIMESTAMP WITHOUT TIMEZONE.
    This function takes a timestamp which is timezone-agnostic, and interprets
    it as a timestamp in UTC, and renders that timestamp as a timestamp in the
    given time zone."""

    def __init__(self, tz):
        self.tz = tz

    def execute(self, col: str) -> Column:
        return F.from_utc_timestamp(col, self.tz)


class GetJsonObject(ColumnPipelineOperation):
    """Extracts json object from a json string based on json path specified, and
    returns json string of the extracted json object. It will return null if the
    input json string is invalid."""

    def __init__(self, path):
        self.path = path

    def execute(self, col: str) -> Column:
        return F.get_json_object(col, self.path)


class Greatest(ColumnPipelineOperation):
    """Returns the greatest value of the list of column names, skipping null values.
    This function takes at least 2 parameters. It will return null iff all
    parameters are null."""

    def __init__(self, *cols):
        self.cols = cols

    def execute(self, _) -> Column:
        return F.greatest(*self.cols)


class Grouping(ColumnPipelineOperation):
    """Aggregate function: indicates whether a specified column in a GROUP BY list is
    aggregated or not, returns 1 for aggregated or 0 for not aggregated in the
    result set."""

    def execute(self, col: str) -> Column:
        return F.grouping(col)


class GroupingId(ColumnPipelineOperation):
    """Aggregate function: returns the level of grouping, equals to"""

    def __init__(self, *cols):
        self.cols = cols

    def execute(self, _) -> Column:
        return F.grouping_id(*self.cols)


class Hash(ColumnPipelineOperation):
    """Calculates the hash code of given columns, and returns the result as an int
    column."""

    def __init__(self, *cols):
        self.cols = cols

    def execute(self, _) -> Column:
        return F.hash(*self.cols)


class Hex(ColumnPipelineOperation):
    """Computes hex value of the given column, which could be
    :class:`pyspark.sql.types.StringType`,
    :class:`pyspark.sql.types.BinaryType`,
    :class:`pyspark.sql.types.IntegerType` or
    :class:`pyspark.sql.types.LongType`."""

    def execute(self, col: str) -> Column:
        return F.hex(col)


class Hour(ColumnPipelineOperation):
    """Extract the hours of a given date as integer."""

    def execute(self, col: str) -> Column:
        return F.hour(col)


class Hypot(ColumnPipelineOperation):
    """Computes ``sqrt(a^2 + b^2)`` without intermediate overflow or underflow."""

    def __init__(self, col2):
        self.col2 = col2

    def execute(self, col: str) -> Column:
        return F.hypot(col, self.col2)


class Initcap(ColumnPipelineOperation):
    """Translate the first letter of each word to upper case in the sentence."""

    def execute(self, col: str) -> Column:
        return F.initcap(col)


class InputFileName(ColumnPipelineOperation):
    """Creates a string column for the file name of the current Spark task."""

    def execute(self, _) -> Column:
        return F.input_file_name()


class Instr(ColumnPipelineOperation):
    """Locate the position of the first occurrence of substr column in the given
    string. Returns null if either of the arguments are null."""

    def __init__(self, substr):
        self.substr = substr

    def execute(self, col: str) -> Column:
        return F.instr(col, self.substr)


class IsNan(ColumnPipelineOperation):
    """An expression that returns true iff the column is NaN."""

    def execute(self, col: str) -> Column:
        return F.isnan(col)


class IsNull(ColumnPipelineOperation):
    """An expression that returns true iff the column is null."""

    def execute(self, col: str) -> Column:
        return F.isnull(col)


class JsonTuple(ColumnPipelineOperation):
    """Creates a new row for a json column according to the given field names."""

    def __init__(self, *fields):
        self.fields = fields

    def execute(self, col: str) -> Column:
        return F.json_tuple(col, *self.fields)


class Kurtosis(ColumnPipelineOperation):
    """Aggregate function: returns the kurtosis of the values in a group."""

    def execute(self, col: str) -> Column:
        return F.kurtosis(col)


class Lag(ColumnPipelineOperation):
    """Window function: returns the value that is `offset` rows before the current row,
    and `default` if there is less than `offset` rows before the current row.
    For example, an `offset` of one will return the previous row at any given
    point in the window partition."""

    def __init__(self, offset=1, default=None):
        self.offset = offset
        self.default = default

    def execute(self, col: str) -> Column:
        return F.lag(col, self.offset, self.default)


class Last(ColumnPipelineOperation):
    """Aggregate function: returns the last value in a group."""

    def __init__(self, ignorenulls=False):
        self.ignorenulls = ignorenulls

    def execute(self, col: str) -> Column:
        return F.last(col, self.ignorenulls)


class LastDay(ColumnPipelineOperation):
    """Returns the last day of the month which the given date belongs to."""

    def execute(self, col: str) -> Column:
        return F.last_day(col)


class Lead(ColumnPipelineOperation):
    """Window function: returns the value that is `offset` rows after the current row,
    and `default` if there is less than `offset` rows after the current row. For
    example, an `offset` of one will return the next row at any given point in
    the window partition."""

    def __init__(self, offset=1, default=None):
        self.offset = offset
        self.default = default

    def execute(self, col: str) -> Column:
        return F.lead(col, self.offset, self.default)


class Least(ColumnPipelineOperation):
    """Returns the least value of the list of column names, skipping null values. This
    function takes at least 2 parameters. It will return null iff all parameters
    are null."""

    def __init__(self, *cols):
        self.cols = cols

    def execute(self, _) -> Column:
        return F.least(*self.cols)


class Length(ColumnPipelineOperation):
    """Computes the character length of string data or number of bytes of binary data.
    The length of character data includes the trailing spaces. The length of
    binary data includes binary zeros."""

    def execute(self, col: str) -> Column:
        return F.length(col)


class Levenshtein(ColumnPipelineOperation):
    """Computes the Levenshtein distance of the two given strings."""

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def execute(self, _) -> Column:
        return F.levenshtein(self.left, self.right)


class Lit(ColumnPipelineOperation):
    """Creates a :class:`~pyspark.sql.Column` of literal value."""

    def __init__(self, value):
        self.value = value

    def execute(self, _) -> Column:
        return F.lit(self.value)


class Locate(ColumnPipelineOperation):
    """Locate the position of the first occurrence of substr in a string column, after
    position pos."""

    def __init__(self, substr, str, pos=1):
        self.substr = substr
        self.str = str
        self.pos = pos

    def execute(self, _) -> Column:
        return F.locate(self.substr, self.str, self.pos)


class Log(ColumnPipelineOperation):
    """Returns the first argument-based logarithm of the second argument."""

    def __init__(self, base=None):
        self.base = base

    def execute(self, col: str) -> Column:
        if not self.base:
            return F.log(col)
        return F.log(self.base, col)


class Log10(ColumnPipelineOperation):
    """Computes the logarithm of the given value in Base 10."""

    def execute(self, col: str) -> Column:
        return F.log10(col)


class Log1P(ColumnPipelineOperation):
    """Computes the natural logarithm of the given value plus one."""

    def execute(self, col: str) -> Column:
        return F.log1p(col)


class Log2(ColumnPipelineOperation):
    """Returns the base-2 logarithm of the argument."""

    def execute(self, col: str) -> Column:
        return F.log2(col)


class Lower(ColumnPipelineOperation):
    """Converts a string expression to lower case."""

    def execute(self, col: str) -> Column:
        return F.lower(col)


class Lpad(ColumnPipelineOperation):
    """Left-pad the string column to width `len` with `pad`."""

    def __init__(self, len, pad):
        self.len = len
        self.pad = pad

    def execute(self, col: str) -> Column:
        return F.lpad(col, self.len, self.pad)


class Ltrim(ColumnPipelineOperation):
    """Trim the spaces from left end for the specified string value."""

    def execute(self, col: str) -> Column:
        return F.ltrim(col)


class MapConcat(ColumnPipelineOperation):
    """Returns the union of all the given maps."""

    def __init__(self, *cols):
        self.cols = cols

    def execute(self, _) -> Column:
        return F.map_concat(*self.cols)


class MapEntries(ColumnPipelineOperation):
    """Collection function: Returns an unordered array of all entries in the given map."""

    def execute(self, col: str) -> Column:
        return F.map_entries(col)


class MapFilter(ColumnPipelineOperation):
    """Returns a map whose key-value pairs satisfy a predicate."""

    def __init__(self, f):
        self.f = f

    def execute(self, col: str) -> Column:
        return F.map_filter(col, self.f)


class MapFromArrays(ColumnPipelineOperation):
    """Creates a new map from two arrays."""

    def __init__(self, col2):
        self.col2 = col2

    def execute(self, col: str) -> Column:
        return F.map_from_arrays(col, self.col2)


class MapFromEntries(ColumnPipelineOperation):
    """Collection function: Returns a map created from the given array of entries."""

    def execute(self, col: str) -> Column:
        return F.map_from_entries(col)


class MapKeys(ColumnPipelineOperation):
    """Collection function: Returns an unordered array containing the keys of the map."""

    def execute(self, col: str) -> Column:
        return F.map_keys(col)


class MapValues(ColumnPipelineOperation):
    """Collection function: Returns an unordered array containing the values of the
    map."""

    def execute(self, col: str) -> Column:
        return F.map_values(col)


class MapZipWith(ColumnPipelineOperation):
    """Merge two given maps, key-wise into a single map using a function."""

    def __init__(self, col2, f):
        self.col2 = col2
        self.f = f

    def execute(self, col: str) -> Column:
        return F.map_zip_with(col, self.col2, self.f)


class Max(ColumnPipelineOperation):
    """Aggregate function: returns the maximum value of the expression in a group."""

    def execute(self, col: str) -> Column:
        return F.max(col)


class Md5(ColumnPipelineOperation):
    """Calculates the MD5 digest and returns the value as a 32 character hex string."""

    def execute(self, col: str) -> Column:
        return F.md5(col)


class Mean(ColumnPipelineOperation):
    """Aggregate function: returns the average of the values in a group."""

    def execute(self, col: str) -> Column:
        return F.mean(col)


class Min(ColumnPipelineOperation):
    """Aggregate function: returns the minimum value of the expression in a group."""

    def execute(self, col: str) -> Column:
        return F.min(col)


class Minute(ColumnPipelineOperation):
    """Extract the minutes of a given date as integer."""

    def execute(self, col: str) -> Column:
        return F.minute(col)


class MonotonicallyIncreasingId(ColumnPipelineOperation):
    """A column that generates monotonically increasing 64-bit integers."""

    def execute(self, _) -> Column:
        return F.monotonically_increasing_id()


class Month(ColumnPipelineOperation):
    """Extract the month of a given date as integer."""

    def execute(self, col: str) -> Column:
        return F.month(col)


class MonthsBetween(ColumnPipelineOperation):
    """Returns number of months between dates date1 and date2. If date1 is later than
    date2, then the result is positive. If date1 and date2 are on the same day
    of month, or both are the last day of month, returns an integer (time of day
    will be ignored). The result is rounded off to 8 digits unless `roundOff` is
    set to `False`."""

    def __init__(self, date1, date2, round_off=True):
        self.date1 = date1
        self.date2 = date2
        self.round_off = round_off

    def execute(self, _) -> Column:
        return F.months_between(self.date1, self.date2, self.round_off)


class Nanvl(ColumnPipelineOperation):
    """Returns col1 if it is not NaN, or col2 if col1 is NaN."""

    def __init__(self, col2):
        self.col2 = col2

    def execute(self, col: str) -> Column:
        return F.nanvl(col, self.col2)


class NextDay(ColumnPipelineOperation):
    """Returns the first date which is later than the value of the date column."""

    def __init__(self, day_of_week):
        self.day_of_week = day_of_week

    def execute(self, col: str) -> Column:
        return F.next_day(col, self.day_of_week)


class NthValue(ColumnPipelineOperation):
    """Window function: returns the value that is the `offset`\th row of the window
    frame (counting from 1), and `null` if the size of window frame is less than
    `offset` rows."""

    def __init__(self, offset, ignore_nulls=False):
        self.offset = offset
        self.ignore_nulls = ignore_nulls

    def execute(self, col: str) -> Column:
        return F.nth_value(col, self.offset, self.ignore_nulls)


class Ntile(ColumnPipelineOperation):
    """Window function: returns the ntile group id (from 1 to `n` inclusive) in an
    ordered window partition. For example, if `n` is 4, the first quarter of the
    rows will get value 1, the second quarter will get 2, the third quarter will
    get 3, and the last quarter will get 4."""

    def __init__(self, n):
        self.n = n

    def execute(self, _) -> Column:
        return F.ntile(self.n)


class Overlay(ColumnPipelineOperation):
    """Overlay the specified portion of `src` with `replace`, starting from byte
    position `pos` of `src` and proceeding for `len` bytes."""

    def __init__(self, replace, pos, len=-1):
        self.replace = replace
        self.pos = pos
        self.len = len

    def execute(self, col: str) -> Column:
        return F.overlay(col, self.replace, self.pos, self.len)


class PercentRank(ColumnPipelineOperation):
    """Window function: returns the relative rank (i.e. percentile) of rows within a
    window partition."""

    def execute(self, _) -> Column:
        return F.percent_rank()


class PercentileApprox(ColumnPipelineOperation):
    """Returns the approximate `percentile` of the numeric column `col` which is the
    smallest value in the ordered `col` values (sorted from least to greatest)
    such that no more than `percentage` of `col` values is less than the value
    or equal to that value. The value of percentage must be between 0.0 and 1.0."""

    def __init__(self, percentage, accuracy=10000):
        self.percentage = percentage
        self.accuracy = accuracy

    def execute(self, col: str) -> Column:
        return F.percentile_approx(col, self.percentage, self.accuracy)


class Posexplode(ColumnPipelineOperation):
    """Returns a new row for each element with position in the given array or map. Uses
    the default column name `pos` for position, and `col` for elements in the
    array and `key` and `value` for elements in the map unless specified
    otherwise."""

    def execute(self, col: str) -> Column:
        return F.posexplode(col)


class PosexplodeOuter(ColumnPipelineOperation):
    """Returns a new row for each element with position in the given array or map.
    Unlike posexplode, if the array/map is null or empty then the row (null,
    null) is produced. Uses the default column name `pos` for position, and
    `col` for elements in the array and `key` and `value` for elements in the
    map unless specified otherwise."""

    def execute(self, col: str) -> Column:
        return F.posexplode_outer(col)


class Pow(ColumnPipelineOperation):
    """Returns the value of the first argument raised to the power of the second
    argument."""

    def __init__(self, col2):
        self.col2 = col2

    def execute(self, col: str) -> Column:
        return F.pow(col, self.col2)


class Product(ColumnPipelineOperation):
    """Aggregate function: returns the product of the values in a group."""

    def execute(self, col: str) -> Column:
        return F.product(col)


class Quarter(ColumnPipelineOperation):
    """Extract the quarter of a given date as integer."""

    def execute(self, col: str) -> Column:
        return F.quarter(col)


class Radians(ColumnPipelineOperation):
    """Converts an angle measured in degrees to an approximately equivalent angle
    measured in radians."""

    def execute(self, col: str) -> Column:
        return F.radians(col)


class RaiseError(ColumnPipelineOperation):
    """Throws an exception with the provided error message."""

    def __init__(self, err_msg):
        self.err_msg = err_msg

    def execute(self, _) -> Column:
        return F.raise_error(self.err_msg)


class Rand(ColumnPipelineOperation):
    """Generates a random column with independent and identically distributed (i.i.d.)
    samples uniformly distributed in [0.0, 1.0)."""

    def __init__(self, seed=None):
        self.seed = seed

    def execute(self, _) -> Column:
        return F.rand(self.seed)


class Randn(ColumnPipelineOperation):
    """Generates a column with independent and identically distributed (i.i.d.) samples
    from the standard normal distribution."""

    def __init__(self, seed=None):
        self.seed = seed

    def execute(self, _) -> Column:
        return F.randn(self.seed)


class Rank(ColumnPipelineOperation):
    """Window function: returns the rank of rows within a window partition."""

    def execute(self, _) -> Column:
        return F.rank()


class RegexpExtract(ColumnPipelineOperation):
    """Extract a specific group matched by a Java regex, from the specified string
    column. If the regex did not match, or the specified group did not match, an
    empty string is returned."""

    def __init__(self, pattern, idx):
        self.pattern = pattern
        self.idx = idx

    def execute(self, col: str) -> Column:
        return F.regexp_extract(col, self.pattern, self.idx)


class RegexpReplace(ColumnPipelineOperation):
    """Replace all substrings of the specified string value that match regexp with rep."""

    def __init__(self, pattern, replacement):
        self.pattern = pattern
        self.replacement = replacement

    def execute(self, col: str) -> Column:
        return F.regexp_replace(col, self.pattern, self.replacement)


class Repeat(ColumnPipelineOperation):
    """Repeats a string column n times, and returns it as a new string column."""

    def __init__(self, n):
        self.n = n

    def execute(self, col: str) -> Column:
        return F.repeat(col, self.n)


class Reverse(ColumnPipelineOperation):
    """Collection function: returns a reversed string or an array with reverse order of
    elements."""

    def execute(self, col: str) -> Column:
        return F.reverse(col)


class Rint(ColumnPipelineOperation):
    """Returns the double value that is closest in value to the argument and is equal
    to a mathematical integer."""

    def execute(self, col: str) -> Column:
        return F.rint(col)


class Round(ColumnPipelineOperation):
    """Round the given value to `scale` decimal places using HALF_UP rounding mode if
    `scale` >= 0 or at integral part when `scale` < 0."""

    def __init__(self, scale=0):
        self.scale = scale

    def execute(self, col: str) -> Column:
        return F.round(col, self.scale)


class RowNumber(ColumnPipelineOperation):
    """Window function: returns a sequential number starting at 1 within a window
    partition."""

    def execute(self, _) -> Column:
        return F.row_number()


class Rpad(ColumnPipelineOperation):
    """Right-pad the string column to width `len` with `pad`."""

    def __init__(self, len, pad):
        self.len = len
        self.pad = pad

    def execute(self, col: str) -> Column:
        return F.rpad(col, self.len, self.pad)


class Rtrim(ColumnPipelineOperation):
    """Trim the spaces from right end for the specified string value."""

    def execute(self, col: str) -> Column:
        return F.rtrim(col)


class SchemaOfCsv(ColumnPipelineOperation):
    """Parses a CSV string and infers its schema in DDL format."""

    def __init__(self, csv, options=None):
        self.csv = csv
        self.options = options

    def execute(self, _) -> Column:
        return F.schema_of_csv(self.csv, self.options)


class SchemaOfJson(ColumnPipelineOperation):
    """Parses a JSON string and infers its schema in DDL format."""

    def __init__(self, json, options=None):
        self.json = json
        self.options = options

    def execute(self, _) -> Column:
        return F.schema_of_json(self.json, self.options)


class Second(ColumnPipelineOperation):
    """Extract the seconds of a given date as integer."""

    def execute(self, col: str) -> Column:
        return F.second(col)


class Sentences(ColumnPipelineOperation):
    """Splits a string into arrays of sentences, where each sentence is an array of
    words. The 'language' and 'country' arguments are optional, and if omitted,
    the default locale is used."""

    def __init__(self, language=None, country=None):
        self.language = language
        self.country = country

    def execute(self, col: str) -> Column:
        return F.sentences(col, self.language, self.country)


class Sequence(ColumnPipelineOperation):
    """Generate a sequence of integers from `start` to `stop`, incrementing by `step`.
    If `step` is not set, incrementing by 1 if `start` is less than or equal to
    `stop`, otherwise -1."""

    def __init__(self, stop, step=None):
        self.stop = stop
        self.step = step

    def execute(self, col: str) -> Column:
        return F.sequence(col, self.stop, self.step)


class SessionWindow(ColumnPipelineOperation):
    """Generates session window given a timestamp specifying column. Session window is
    one of dynamic windows, which means the length of window is varying
    according to the given inputs. The length of session window is defined as
    "the timestamp of latest input of the session + gap duration", so when the
    new inputs are bound to the current session window, the end time of session
    window can be expanded according to the new inputs. Windows can support
    microsecond precision. Windows in the order of months are not supported. For
    a streaming query, you may use the function `current_timestamp` to generate
    windows on processing time. gapDuration is provided as strings, e.g. '1
    second', '1 day 12 hours', '2 minutes'. Valid interval strings are 'week',
    'day', 'hour', 'minute', 'second', 'millisecond', 'microsecond'. It could
    also be a Column which can be evaluated to gap duration dynamically based on
    the input row. The output column will be a struct called 'session_window' by
    default with the nested columns 'start' and 'end', where 'start' and 'end'
    will be of :class:`pyspark.sql.types.TimestampType`."""

    def __init__(self, gap_duration):
        self.gap_duration = gap_duration

    def execute(self, col: str) -> Column:
        return F.session_window(col, self.gap_duration)


class Sha1(ColumnPipelineOperation):
    """Returns the hex string result of SHA-1."""

    def execute(self, col: str) -> Column:
        return F.sha1(col)


class Sha2(ColumnPipelineOperation):
    """Returns the hex string result of SHA-2 family of hash functions (SHA-224,
    SHA-256, SHA-384, and SHA-512). The numBits indicates the desired bit length
    of the result, which must have a value of 224, 256, 384, 512, or 0 (which is
    equivalent to 256)."""

    def __init__(self, num_bits):
        self.num_bits = num_bits

    def execute(self, col: str) -> Column:
        return F.sha2(col, self.num_bits)


class Shiftleft(ColumnPipelineOperation):
    """Shift the given value numBits left."""

    def __init__(self, num_bits):
        self.num_bits = num_bits

    def execute(self, col: str) -> Column:
        return F.shiftleft(col, self.num_bits)


class Shiftright(ColumnPipelineOperation):
    """(Signed) shift the given value numBits right."""

    def __init__(self, num_bits):
        self.num_bits = num_bits

    def execute(self, col: str) -> Column:
        return F.shiftright(col, self.num_bits)


class Shiftrightunsigned(ColumnPipelineOperation):
    """Unsigned shift the given value numBits right."""

    def __init__(self, num_bits):
        self.num_bits = num_bits

    def execute(self, col: str) -> Column:
        return F.shiftrightunsigned(col, self.num_bits)


class Shuffle(ColumnPipelineOperation):
    """Collection function: Generates a random permutation of the given array."""

    def execute(self, col: str) -> Column:
        return F.shuffle(col)


class Signum(ColumnPipelineOperation):
    """Computes the signum of the given value."""

    def execute(self, col: str) -> Column:
        return F.signum(col)


class Sin(ColumnPipelineOperation):
    """sine of the angle, as if computed by `java.lang.Math.sin()`"""

    def execute(self, col: str) -> Column:
        return F.sin(col)


class Sinh(ColumnPipelineOperation):
    """hyperbolic sine of the given value, as if computed by `java.lang.Math.sinh()`"""

    def execute(self, col: str) -> Column:
        return F.sinh(col)


class Size(ColumnPipelineOperation):
    """Collection function: returns the length of the array or map stored in the
    column."""

    def execute(self, col: str) -> Column:
        return F.size(col)


class Skewness(ColumnPipelineOperation):
    """Aggregate function: returns the skewness of the values in a group."""

    def execute(self, col: str) -> Column:
        return F.skewness(col)


class Slice(ColumnPipelineOperation):
    """Collection function: returns an array containing  all the elements in `x` from
    index `start` (array indices start at 1, or from the end if `start` is
    negative) with the specified `length`."""

    def __init__(self, x, length):
        self.x = x
        self.length = length

    def execute(self, col: str) -> Column:
        return F.slice(col, self.x, self.length)


class SortArray(ColumnPipelineOperation):
    """Collection function: sorts the input array in ascending or descending order
    according to the natural ordering of the array elements. Null elements will
    be placed at the beginning of the returned array in ascending order or at
    the end of the returned array in descending order."""

    def __init__(self, asc=True):
        self.asc = asc

    def execute(self, col: str) -> Column:
        return F.sort_array(col, self.asc)


class Soundex(ColumnPipelineOperation):
    """Returns the SoundEx encoding for a string"""

    def execute(self, col: str) -> Column:
        return F.soundex(col)


class SparkPartitionId(ColumnPipelineOperation):
    """A column for partition ID."""

    def execute(self, _) -> Column:
        return F.spark_partition_id()


class Split(ColumnPipelineOperation):
    """Splits str around matches of the given pattern."""

    def __init__(self, pattern, limit=-1):
        self.pattern = pattern
        self.limit = limit

    def execute(self, col: str) -> Column:
        return F.split(col, self.pattern, self.limit)


class Sqrt(ColumnPipelineOperation):
    """Computes the square root of the specified float value."""

    def execute(self, col: str) -> Column:
        return F.sqrt(col)


class Stddev(ColumnPipelineOperation):
    """Aggregate function: alias for stddev_samp."""

    def execute(self, col: str) -> Column:
        return F.stddev(col)


class StddevPop(ColumnPipelineOperation):
    """Aggregate function: returns population standard deviation of the expression in a
    group."""

    def execute(self, col: str) -> Column:
        return F.stddev_pop(col)


class StddevSamp(ColumnPipelineOperation):
    """Aggregate function: returns the unbiased sample standard deviation of the
    expression in a group."""

    def execute(self, col: str) -> Column:
        return F.stddev_samp(col)


class Struct(ColumnPipelineOperation):
    """Creates a new struct column."""

    def __init__(self, *cols):
        self.cols = cols

    def execute(self, _) -> Column:
        return F.struct(*self.cols)


class Substring(ColumnPipelineOperation):
    """Substring starts at `pos` and is of length `len` when str is String type or
    returns the slice of byte array that starts at `pos` in byte and is of
    length `len` when str is Binary type."""

    def __init__(self, pos, len):
        self.pos = pos
        self.len = len

    def execute(self, col: str) -> Column:
        return F.substring(col, self.pos, self.len)


class SubstringIndex(ColumnPipelineOperation):
    """Returns the substring from string str before count occurrences of the delimiter
    delim. If count is positive, everything the left of the final delimiter
    (counting from left) is returned. If count is negative, every to the right
    of the final delimiter (counting from the right) is returned.
    substring_index performs a case-sensitive match when searching for delim."""

    def __init__(self, delim, count):
        self.delim = delim
        self.count = count

    def execute(self, col: str) -> Column:
        return F.substring_index(col, self.delim, self.count)


class Sum(ColumnPipelineOperation):
    """Aggregate function: returns the sum of all values in the expression."""

    def execute(self, col: str) -> Column:
        return F.sum(col)


class SumDistinct(ColumnPipelineOperation):
    """Aggregate function: returns the sum of distinct values in the expression."""

    def execute(self, col: str) -> Column:
        return F.sum_distinct(col)


class Tan(ColumnPipelineOperation):
    """tangent of the given value, as if computed by `java.lang.Math.tan()`"""

    def execute(self, col: str) -> Column:
        return F.tan(col)


class Tanh(ColumnPipelineOperation):
    """hyperbolic tangent of the given value as if computed by `java.lang.Math.tanh()`"""

    def execute(self, col: str) -> Column:
        return F.tanh(col)


class TimestampSeconds(ColumnPipelineOperation):
    """Gets a timestamp truncated at seconds"""

    def execute(self, col: str) -> Column:
        return F.timestamp_seconds(col)


class ToCsv(ColumnPipelineOperation):
    """Converts a column containing a :class:`StructType` into a CSV string. Throws an
    exception, in the case of an unsupported type."""

    def __init__(self, options=None):
        self.options = options

    def execute(self, col: str) -> Column:
        return F.to_csv(col, self.options)


class ToDate(ColumnPipelineOperation):
    """Converts a :class:`~pyspark.sql.Column` into :class:`pyspark.sql.types.DateType`
    using the optionally specified format. Specify formats according to
    `datetime pattern`_. By default, it follows casting rules to
    :class:`pyspark.sql.types.DateType` if the format is omitted. Equivalent to
    ``col.cast("date")``."""

    def __init__(self, format=None):
        self.format = format

    def execute(self, col: str) -> Column:
        return F.to_date(col, self.format)


class ToJson(ColumnPipelineOperation):
    """Converts a column containing a :class:`StructType`, :class:`ArrayType` or a
    :class:`MapType` into a JSON string. Throws an exception, in the case of an
    unsupported type."""

    def __init__(self, options=None):
        self.options = options

    def execute(self, col: str) -> Column:
        return F.to_json(col, self.options)


class ToTimestamp(ColumnPipelineOperation):
    """Converts a :class:`~pyspark.sql.Column` into
    :class:`pyspark.sql.types.TimestampType` using the optionally specified
    format. Specify formats according to `datetime pattern`_. By default, it
    follows casting rules to :class:`pyspark.sql.types.TimestampType` if the
    format is omitted. Equivalent to ``col.cast("timestamp")``."""

    def __init__(self, format=None):
        self.format = format

    def execute(self, col: str) -> Column:
        return F.to_timestamp(col, self.format)


class ToUtcTimestamp(ColumnPipelineOperation):
    """This is a common function for databases supporting TIMESTAMP WITHOUT TIMEZONE.
    This function takes a timestamp which is timezone-agnostic, and interprets
    it as a timestamp in the given timezone, and renders that timestamp as a
    timestamp in UTC."""

    def __init__(self, tz):
        self.tz = tz

    def execute(self, col: str) -> Column:
        return F.to_utc_timestamp(col, self.tz)


class Transform(ColumnPipelineOperation):
    """Returns an array of elements after applying a transformation to each element in
    the input array."""

    def __init__(self, f):
        self.f = f

    def execute(self, col: str) -> Column:
        return F.transform(col, self.f)


class TransformKeys(ColumnPipelineOperation):
    """Applies a function to every key-value pair in a map and returns a map with the
    results of those applications as the new keys for the pairs."""

    def __init__(self, f):
        self.f = f

    def execute(self, col: str) -> Column:
        return F.transform_keys(col, self.f)


class TransformValues(ColumnPipelineOperation):
    """Applies a function to every key-value pair in a map and returns a map with the
    results of those applications as the new values for the pairs."""

    def __init__(self, f):
        self.f = f

    def execute(self, col: str) -> Column:
        return F.transform_values(col, self.f)


class Translate(ColumnPipelineOperation):
    """A function translate any character in the `srcCol` by a character in `matching`.
    The characters in `replace` is corresponding to the characters in
    `matching`. The translate will happen when any character in the string
    matching with the character in the `matching`."""

    def __init__(self, matching, replace):
        self.matching = matching
        self.replace = replace

    def execute(self, col: str) -> Column:
        return F.translate(col, self.matching, self.replace)


class Trim(ColumnPipelineOperation):
    """Trim the spaces from both ends for the specified string column."""

    def execute(self, col: str) -> Column:
        return F.trim(col)


class Trunc(ColumnPipelineOperation):
    """Returns date truncated to the unit specified by the format."""

    def __init__(self, format):
        self.format = format

    def execute(self, col: str) -> Column:
        return F.trunc(col, self.format)


class Unbase64(ColumnPipelineOperation):
    """Decodes a BASE64 encoded string column and returns it as a binary column."""

    def execute(self, col: str) -> Column:
        return F.unbase64(col)


class Unhex(ColumnPipelineOperation):
    """Inverse of hex. Interprets each pair of characters as a hexadecimal number and
    converts to the byte representation of number."""

    def execute(self, col: str) -> Column:
        return F.unhex(col)


class UnixTimestamp(ColumnPipelineOperation):
    """Convert time string with given pattern ('yyyy-MM-dd HH:mm:ss', by default) to
    Unix time stamp (in seconds), using the default timezone and the default
    locale, return null if fail."""

    def __init__(self, timestamp=None, format="yyyy-MM-dd HH:mm:ss"):
        self.timestamp = timestamp
        self.format = format

    def execute(self, _) -> Column:
        return F.unix_timestamp(self.timestamp, self.format)


class Upper(ColumnPipelineOperation):
    """Converts a string expression to upper case."""

    def execute(self, col: str) -> Column:
        return F.upper(col)


class VarPop(ColumnPipelineOperation):
    """Aggregate function: returns the population variance of the values in a group."""

    def execute(self, col: str) -> Column:
        return F.var_pop(col)


class VarSamp(ColumnPipelineOperation):
    """Aggregate function: returns the unbiased sample variance of the values in a
    group."""

    def execute(self, col: str) -> Column:
        return F.var_samp(col)


class Variance(ColumnPipelineOperation):
    """Aggregate function: alias for var_samp"""

    def execute(self, col: str) -> Column:
        return F.variance(col)


class WeekOfYear(ColumnPipelineOperation):
    """Extract the week number of a given date as integer."""

    def execute(self, col: str) -> Column:
        return F.weekofyear(col)


class When(ColumnPipelineOperation):
    """Evaluates a list of conditions and returns one of multiple possible result
    expressions. If :func:`pyspark.sql.Column.otherwise` is not invoked, None is
    returned for unmatched conditions."""

    def __init__(self, condition, value):
        self.condition = condition
        self.value = value

    def execute(self, _) -> Column:
        return F.when(self.condition, self.value)


class Window(ColumnPipelineOperation):
    """Bucketize rows into one or more time windows given a timestamp specifying
    column. Window starts are inclusive but the window ends are exclusive, e.g.
    12:05 will be in the window [12:05,12:10) but not in [12:00,12:05). Windows
    can support microsecond precision. Windows in the order of months are not
    supported."""

    def __init__(self, window_duration, slide_duration=None, start_time=None):
        self.window_duration = window_duration
        self.slide_duration = slide_duration
        self.start_time = start_time

    def execute(self, col: str) -> Column:
        return F.window(col, self.window_duration, self.slide_duration, self.start_time)


class Xxhash64(ColumnPipelineOperation):
    """Calculates the hash code of given columns using the 64-bit variant of the xxHash
    algorithm, and returns the result as a long column."""

    def __init__(self, *cols):
        self.cols = cols

    def execute(self, _) -> Column:
        return F.xxhash64(*self.cols)


class Year(ColumnPipelineOperation):
    """Extract the year of a given date as integer."""

    def execute(self, col: str) -> Column:
        return F.year(col)


class ZipWith(ColumnPipelineOperation):
    """Merge two given arrays, element-wise, into a single array using a function. If
    one array is shorter, nulls are appended at the end to match the length of
    the longer array, before applying the function."""

    def __init__(self, left, right, f):
        self.left = left
        self.right = right
        self.f = f

    def execute(self, _) -> Column:
        return F.zip_with(self.left, self.right, self.f)
