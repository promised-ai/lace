import numpy as np
import polars as pl

import lace


def test_engine_from_df_with_codebook_smoke():
    n = 14
    df = pl.DataFrame(
        {
            "ID": list(range(n)),
            "x": np.random.randn(14),
            "y": np.random.randint(2, size=n),
        }
    )
    codebook = lace.core.codebook_from_df(df)
    engine = lace.Engine.from_df(df, codebook=codebook, n_states=3)
    assert engine.shape == (14, 2)
    assert engine.columns == ["x", "y"]
