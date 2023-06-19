import numpy as np
import pandas as pd
import polars as pl
import pytest

from lace.examples import Satellites


@pytest.fixture()
def satellites():
    satellites = Satellites()
    return satellites


def test_append_pandas_df_no_metadata(satellites):
    n_rows = satellites.shape[0]
    data = np.random.rand(n_rows)
    df = pd.DataFrame(
        {
            "new_col": data,
        },
        index=satellites.index,
    )
    satellites.append_columns(df)

    for i, x in enumerate(data):
        assert satellites[i, "new_col"] == pytest.approx(x)


def test_append_polars_df_no_metadata(satellites):
    n_rows = satellites.shape[0]
    data = np.random.rand(n_rows)
    df = pl.DataFrame(
        {
            "new_col": data,
            "index": satellites.index,
        }
    )
    satellites.append_columns(df)

    for i, x in enumerate(data):
        assert satellites[i, "new_col"] == pytest.approx(x)


def test_append_polars_df_no_metadata_sparse(satellites):
    n_rows = satellites.shape[0]
    ixs = np.random.choice(satellites.index, size=n_rows // 2, replace=False)
    data = np.random.rand(n_rows // 2)
    df = pl.DataFrame({"new_col": data, "index": ixs})
    satellites.append_columns(df)

    for ix, x in zip(ixs, data):
        assert satellites[ix, "new_col"] == pytest.approx(x)


def test_append_pandas_df_no_metadata_sparse(satellites):
    n_rows = satellites.shape[0]
    ixs = np.random.choice(satellites.index, size=n_rows // 2, replace=False)
    data = np.random.rand(n_rows // 2)
    df = pd.DataFrame({"new_col": data}, index=ixs)
    satellites.append_columns(df)

    for ix, x in zip(ixs, data):
        assert satellites[ix, "new_col"] == pytest.approx(x)


@pytest.mark.parametrize("choices", [["red", "blue", "green"], [0, 1, 2, 3]])
def test_append_pandas_df_no_metadata_sparse_categorical(satellites, choices):
    n_rows = satellites.shape[0]
    ixs = np.random.choice(satellites.index, size=n_rows // 2, replace=False)
    data = np.random.choice(choices, size=n_rows // 2, replace=True)
    df = pd.DataFrame({"new_col": data}, index=ixs)
    satellites.append_columns(df)

    for ix, x in zip(ixs, data):
        assert satellites[ix, "new_col"] == x


@pytest.mark.parametrize("choices", [["red", "blue", "green"], [0, 1, 2, 3]])
def test_append_polars_df_no_metadata_sparse_categorical(satellites, choices):
    n_rows = satellites.shape[0]
    ixs = np.random.choice(satellites.index, size=n_rows // 2, replace=False)
    data = np.random.choice(choices, size=n_rows // 2, replace=True)
    df = pl.DataFrame({"new_col": data, "index": ixs})
    satellites.append_columns(df)

    for ix, x in zip(ixs, data):
        assert satellites[ix, "new_col"] == x
