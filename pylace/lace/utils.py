import itertools as it
from typing import List, Optional, Union

import pandas as pd
import polars as pl
from scipy.cluster.hierarchy import dendrogram, linkage

from lace import ColumnMetadata
from lace.core import infer_srs_metadata


class Dimension:
    Rows = 0
    Colums = 1


FN_IS_SYMMETRIC = {
    "mi": False,
    "depprob": False,
    "rowsim": False,
}


FN_DIMENSION = {
    "mi": Dimension.Colums,
    "depprob": Dimension.Colums,
    "rowsim": Dimension.Rows,
}


def _diagnostic(name, state_diag):
    if name == "loglike":
        return pl.Series("loglike", state_diag["loglike"])
    elif name == "logprior":
        return pl.Series("logprior", state_diag["logprior"])
    elif name == "score":
        loglike = pl.Series("loglike", state_diag["loglike"])
        logprior = pl.Series("logprior", state_diag["logprior"])
        score = loglike + logprior
        return score.rename("score")
    else:
        raise ValueError(f"Invalid diagnostic: `{name}`")


def get_all_pairs(fn_name, engine):
    if fn_name not in FN_DIMENSION:
        raise ValueError(f"{fn_name} is an invalid pairwise function")

    indices = (
        engine.index
        if FN_DIMENSION[fn_name] == Dimension.Rows
        else engine.columns
    )

    symmetric = FN_IS_SYMMETRIC[fn_name]

    if symmetric:
        pairs = []
        n = len(indices)
        for i in range(n):
            for j in range(i, n):
                pairs.append((indices[i], indices[j]))
    else:
        pairs = list(it.product(indices, indices))

    return pairs, symmetric


def hcluster(df: pl.DataFrame, method="ward"):
    z = linkage(df[:, 1:], method=method, optimal_ordering=True)
    dendro = dendrogram(z, no_plot=True)
    leaves = dendro["leaves"]
    col_ixs = [0] + [i + 1 for i in leaves]
    return df[leaves, col_ixs], z


def return_srs(srs: Optional[Union[pl.Series, float]]):
    if srs is None:
        return None

    if isinstance(srs, float):
        return srs

    n = srs.shape[0]
    if n == 0:
        return None
    elif n == 1:
        return srs[0]
    else:
        return srs


def infer_column_metadata(
    df: Union[pl.DataFrame, pd.DataFrame],
    cat_cutoff: int = 20,
    no_hypers: bool = False,
) -> List[ColumnMetadata]:
    """
    Infer the column metadata from data.

    Parameters
    ----------
    df: pl.DataFrame or pd.DataFrame
        The input data. Columns named "index" or "id" will be ignored
    cat_cutoff: int, optional
        The max value of an unsigned integer a column can have before it is
        inferred to be count type (default: 20)
    no_hypres: bool, optional
        If True, the prior will be fixed and hyper priors will be ignored
    """
    mds = []
    for column in df.columns:
        if column.lower() in ("id", "index"):
            continue
        md = infer_srs_metadata(
            pl.Series(df[column]), cat_cutoff, no_hypers
        ).rename(column)
        mds.append(md)
    return mds
