import doctest
import os
from importlib import import_module

import polars as pl
from plotly import io

HIDE_PLOT = os.environ.get("LACE_DOCTEST_NOPLOT", "0") == "1"


if HIDE_PLOT:
    # io.renderers.default = 'json'
    io.renderers.default = "iframe"


pl.Config(tbl_rows=8)


class _Context(dict):
    def clear(self):
        pass

    def copy(self):
        return self


def plot():
    module = import_module("lace.plot")
    extraglobs = _Context(module.__dict__.copy())

    doctest.testmod(module, extraglobs=extraglobs)


def engine():
    module = import_module("lace.engine")
    extraglobs = _Context(module.__dict__.copy())

    doctest.testmod(module, extraglobs=extraglobs)


def analysis():
    module = import_module("lace.analysis")
    extraglobs = _Context(module.__dict__.copy())

    doctest.testmod(module, extraglobs=extraglobs)


if __name__ == "__main__":
    engine()
    analysis()
    plot()
