#!/usr/bin/env python

# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name

import re
import sys
import textwrap
from argparse import ArgumentParser
from inspect import getmembers, isfunction, signature, _empty, Parameter, _ParameterKind
from pathlib import Path
from typing import IO, Any, Dict, List

from jinja2 import Template
from pyspark.sql import functions


class Attr:
    """Wraps params of original PySpark function to be used as attributes of a
    ColumnPipelineOperation class
    """

    def __init__(
        self, name: str, is_var_arg: bool = False, default_value: Any = _empty
    ):
        self.name = camel_to_snake(name)
        self.is_var_arg = is_var_arg
        self.default_value = default_value

    @classmethod
    def from_parameter(cls, param: Parameter) -> "Attr":
        """Creates a new `Attr` from an `inspect.parameter`"""
        return Attr(
            name=param.name,
            is_var_arg=param.kind == _ParameterKind.VAR_POSITIONAL,
            default_value=param.default,
        )

    def render_as_init_param(self) -> str:
        """Renders the attr as a string to be used as a parameter for `__init__`"""
        splat = "*" if self.is_var_arg else ""
        output = f"{splat}{self.name}"

        if not self.is_var_arg and self.default_value is not _empty:
            output += f"={self.default_value!r}"

        return output

    def render_as_init_assign(self) -> str:
        """Renders the attr as a string to be used as an assignment in `__init__`"""
        return f"self.{self.name} = {self.name}"

    def render_as_exec_arg(self) -> str:
        """Renders the attr as a string to be used as an argument for the internal
        PySpark function call in `execute`
        """
        splat = "*" if self.is_var_arg else ""
        return f"{splat}self.{self.name}"


skip_names = {
    # imported funcs
    "since",
    "pandas_udf",
    "to_str",
    # deprecated funcs
    "sumDistinct",
    "toDegrees",
    "toRadians",
    "bitwiseNOT",
    "approxCountDistinct",
    "shiftLeft",
    "shiftRight",
    "shiftRightUnsigned",
    # df partitioning funcs
    "years",
    "months",
    "days",
    "hours",
    "bucket",
    # others
    "broadcast",
    "column",
    "udf",
}


def camel_to_snake(val: str) -> str:
    """Converts a camel case string to snake case"""
    cap_subbed = re.sub(r"([A-Z])", r"_\1", val)
    of_subbed = re.sub(r"(\w)of(\w)", r"\1_of_\2", cap_subbed)
    is_subbed = re.sub(r"^is(\w)", r"is_\1", of_subbed)
    return is_subbed.lower()


def make_class_name(val: str) -> str:
    """Converts any string to pascal case"""
    snaked = camel_to_snake(val)
    broken = snaked.replace("_", " ")
    titled = broken.title()
    return "".join(x for x in titled if not x.isspace())


def get_docstring(dstring: str) -> str:
    """Parses a PySpark function docstring for the useful bits"""
    lines: List[str] = []

    for line in dstring.splitlines():
        line = line.strip()
        if not line:
            if lines:
                break
            continue
        lines.append(line)

    if lines[0].startswith(".."):
        lines.clear()
        reached_returns = False

        for line in dstring.splitlines():
            line = line.strip()
            if not reached_returns and not line.startswith("Return"):
                continue
            if not reached_returns and line.startswith("Return"):
                reached_returns = True
                continue
            if reached_returns:
                if not line.startswith("-") and not line.startswith(":"):
                    lines.append(line)

    return "\n".join(textwrap.wrap(" ".join(lines), width=80, subsequent_indent="    "))


def parse_pyspark_functions_module() -> Dict[str, str]:
    """Parses the `pyspark.sql.functions` module and creates wrapper
    `ColumnPipelineOperation` objects for the transformation functions.
    """
    fname = (
        Path(__file__).parent.joinpath("templates", "func.py.j2").resolve().as_posix()
    )
    with open(fname, mode="r", encoding="utf-8") as fh:
        template = Template(fh.read())

    classes: Dict[str, str] = {}

    for fn_name, fn in getmembers(functions, isfunction):
        if fn_name in skip_names or fn_name.startswith("_"):
            continue

        sig = signature(fn)
        lensig = len(sig.parameters)
        strsig = str(sig)

        attrs: List[Attr] = []
        no_col = False

        if fn_name == "log":
            if lensig == 1:
                continue
            attrs = [Attr("base")]

        elif fn_name == "lit":
            attrs = [Attr("value")]
            no_col = True

        elif fn_name == "col":
            attrs = [Attr("col")]
            no_col = True

        elif strsig == "(col)" or strsig == "(date)":
            pass

        elif strsig == "(*cols)":
            attrs = [Attr("cols", True)]
            no_col = True

        elif fn_name == "arrays_overlap":
            attrs = [
                Attr("left"),
                Attr("right"),
            ]
            no_col = True

        elif fn_name == "when":
            attrs = [
                Attr("condition"),
                Attr("value"),
            ]
            no_col = True

        elif re.match(
            r"^\(((col\d?)|(date)|(src(Col)?)|(start)|(str(ing)?)|(timeColumn)|(timestamp)|(x))?, ",
            strsig,
        ):
            attrs = [
                Attr.from_parameter(param)
                for pname, param in sig.parameters.items()
                if pname
                not in (
                    "col",
                    "col1",
                    "date",
                    "src",
                    "srcCol",
                    "start",
                    "str",
                    "string",
                    "timeColumn",
                    "timestamp",
                )
            ]

        else:
            attrs = [Attr.from_parameter(param) for param in sig.parameters.values()]
            no_col = True

        eargs = [a.render_as_exec_arg() for a in attrs]
        if no_col:
            execol = "_"
        else:
            execol = "col: str"
            eargs.insert(0, "col")
        exec_args = ", ".join(eargs)

        class_name = make_class_name(fn_name)

        output = template.render(
            class_name=class_name,
            docstring=get_docstring(fn.__doc__),
            attrs=attrs,
            execol=execol,
            fn_name=fn_name,
            exec_args=exec_args,
        )

        classes[class_name] = output

    return classes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", help="Output filepath; defaults to STDOUT")
    args = parser.parse_args()

    if args.output:
        fh: IO = open(args.output, mode="w", encoding="utf-8")
    else:
        fh = sys.stdout

    classes = parse_pyspark_functions_module()

    fh.writelines(
        [
            "from pyspark.sql import Column\n",
            "from pyspark.sql import functions as F\n",
            "\n",
            "from .proto import ColumnPipelineOperation\n",
            "\n",
            "__all__ = [\n",
        ]
    )
    for class_name in classes:
        fh.write(f'    "{class_name}",\n')
    fh.writelines(
        [
            "]\n",
            "\n",
        ]
    )

    for classdef in classes.values():
        fh.write(f"{classdef}\n")
