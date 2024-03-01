import numpy as np
import pandas as pd
import polars as pl

import lace


def test_engine_from_polars_with_codebook_smoke():
    n = 14
    df = pl.DataFrame(
        {
            "ID": list(range(n)),
            "x": np.random.randn(14),
            "y": np.random.randint(2, size=n),
        }
    )
    codebook = lace.Codebook.from_df("test", df)
    assert codebook.shape == (14, 2)

    engine = lace.Engine.from_df(df, codebook=codebook, n_states=3)
    assert engine.shape == (14, 2)
    assert engine.columns == ["x", "y"]


def test_engine_from_pandas_with_codebook_smoke():
    n = 14
    df = pd.DataFrame(
        {
            "x": np.random.randn(14),
            "y": np.random.randint(2, size=n),
        },
        index=list(range(n)),
    )
    df.index.rename("ID", inplace=True)

    codebook = lace.Codebook.from_df("test", df)
    assert codebook.shape == (14, 2)

    engine = lace.Engine.from_df(df, codebook=codebook, n_states=3)
    assert engine.shape == (14, 2)
    assert engine.columns == ["x", "y"]


def test_engine_with_boolean_string_columns():
    n = 14
    df = pl.DataFrame(
        {
            "ID": list(range(n)),
            "x": np.random.randn(n),
            "b": np.random.choice([True, False], size=n),
        }
    )

    assert df.dtypes[df.get_column_index("b")] == pl.Boolean

    codebook = lace.Codebook.from_df("test", df)
    assert codebook.shape == (n, 2)
    assert str(codebook.column_metadata["b"].value_map) == str(
        lace.ValueMap.bool()
    )

    engine = lace.Engine.from_df(df, codebook=codebook, n_states=3)
    assert engine.shape == (n, 2)
    assert engine.columns == ["x", "b"]
    assert str(engine.codebook.column_metadata["b"].value_map) == str(
        lace.ValueMap.bool()
    )
