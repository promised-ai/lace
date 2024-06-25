import doctest
import os
from importlib import import_module

import polars as pl
from plotly import io

NOPLOT = os.environ.get("LACE_DOCTEST_NOPLOT", "0") == "1"


if NOPLOT:
    io.renderers.default = "json"


pl.Config(tbl_rows=8)


class _Context(dict):
    def clear(self):
        pass

    def copy(self):
        return self


def runtest(mod):
    module = import_module(mod)
    extraglobs = _Context(module.__dict__.copy())
    doctest.testmod(module, extraglobs=extraglobs)


if __name__ == "__main__":
    runtest("lace.engine")
    runtest("lace.analysis")
    runtest("lace.codebook")

    if not NOPLOT:
        runtest("lace.plot")
