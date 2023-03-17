import itertools as it
from typing import Optional, Union

import polars as pl
from scipy.cluster.hierarchy import dendrogram, linkage


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
