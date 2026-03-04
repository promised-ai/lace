from math import exp
import itertools as it
from typing import Dict, List, Literal, Optional, Tuple, Union

from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
import polars as pl
from scipy.cluster.hierarchy import dendrogram, linkage

from lace import ColumnMetadata
from lace.core import (
    ColumnKernel,
    RowKernel,
    StateTransition,
    infer_srs_metadata,
)


class Dimension:
    Rows = 0
    Columns = 1


FN_IS_SYMMETRIC = {
    "mi": False,
    "depprob": False,
    "rowsim": False,
}


FN_DIMENSION = {
    "mi": Dimension.Columns,
    "depprob": Dimension.Columns,
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


_COMMON_TRANSITIONS = {
    "sams": [
        StateTransition.view_prior_process_params(),
        StateTransition.row_assignment(RowKernel.sams()),
        StateTransition.view_prior_process_params(),
        StateTransition.row_assignment(RowKernel.sams()),
        StateTransition.view_prior_process_params(),
        StateTransition.row_assignment(RowKernel.slice()),
        StateTransition.component_parameters(),
        StateTransition.column_assignment(ColumnKernel.gibbs()),
        StateTransition.state_prior_process_params(),
        StateTransition.feature_priors(),
    ],
    "flat": [
        StateTransition.view_prior_process_params(),
        StateTransition.row_assignment(RowKernel.sams()),
        StateTransition.view_prior_process_params(),
        StateTransition.row_assignment(RowKernel.sams()),
        StateTransition.view_prior_process_params(),
        StateTransition.component_parameters(),
        StateTransition.row_assignment(RowKernel.slice()),
        StateTransition.component_parameters(),
        StateTransition.view_prior_process_params(),
        StateTransition.feature_priors(),
    ],
    "fast": [
        StateTransition.view_prior_process_params(),
        StateTransition.row_assignment(RowKernel.slice()),
        StateTransition.component_parameters(),
        StateTransition.feature_priors(),
        StateTransition.column_assignment(ColumnKernel.slice()),
        StateTransition.state_prior_process_params(),
    ],
}


def _get_common_transitions(name: str) -> List[StateTransition]:
    transitions = _COMMON_TRANSITIONS.get(name)
    if transitions is None:
        keys_str = ", ".join(_COMMON_TRANSITIONS.keys())
        raise ValueError(
            f"{name} is not a valid transitions set name. valid options are: {keys_str}"
        )
    return transitions


def predict_xs(
    engine, target, given, *, n_points=1_000, mass=0.99
) -> pl.Series:
    ftype = engine.ftype(target)
    if ftype == "Continuous":
        xs = engine.simulate([target], given=given, n=10_000)
        rm = (1.0 - mass) / 2
        a = xs[target].quantile(rm)
        b = xs[target].quantile(1.0 - rm)
        xs = np.linspace(a, b, n_points)
        return pl.Series(target, xs)
    elif ftype == "Categorical":
        return pl.Series(target, engine.engine.categorical_support(target))
    else:
        raise ValueError("unsupported ftype")


def predopt_err(
    engine,
    target: str,
    data: Dict,
    objective: Literal["abserr", "relerr", "sqerr"],
):
    ixs = []
    errs = []
    ftype = engine.ftype(target)
    is_categorical = ftype.lower() == "categorical"

    for ix, row in data.items():
        val = row.pop(target, None)
        if val is None:
            continue
        given = {k: v for k, v in row.items() if v is not None}
        pred = engine.predict(target, given=given, with_uncertainty=False)
        ixs.append(ix)

        row[target] = val

        if is_categorical:
            err = float(val != pred)
        else:
            match objective:
                case "abserr":
                    err = abs(val - pred)
                case "relerr":
                    err = 1.0 - abs(pred / val)
                case "sqerr":
                    err = (val - pred) * (val - pred)
                case _:
                    raise ValueError(f"Unsupported obejctive `{objective}`")

        errs.append(err)

    return ixs, errs


def predopt_unc(
    engine,
    target: str,
    data: Dict,
):
    ixs = []
    errs = []
    for ix, row in data.items():
        given = {k: v for k, v in row.items() if not pd.isnull(v)}
        val = given.pop(target, None)
        if val is None:
            continue
        _, unc = engine.predict(target, given=given, with_uncertainty=True)

        given[target] = val
        ixs.append(ix)
        errs.append(unc)

    return ixs, np.array(errs)


def predopt_objective(
    engine,
    target: str,
    data: Optional[Dict],
    objective: Literal[
        "wrong", "surprial", "neglogp", "abserr", "relerr", "sqerr", "unc"
    ],
) -> Tuple[List, ArrayLike]:
    match objective:
        case "wrong":
            imp = engine.impute(
                target, rows=engine.index, with_uncertainty=False
            )
            wrong = imp[target] != engine[:, target][target]
            return imp["index"], wrong.to_numpy().astype(float)

        case "surprisal":
            surp = engine.surprisal(target)
            return surp["index"].to_list(), np.array(surp["surprisal"].exp())
        case "neglogp":
            assert data is not None
            ixs = []
            ps = []
            for ix, row in data.items():
                given = {k: v for k, v in row.items() if not pd.isnull(v)}
                val = given.pop(target, None)
                if val is None:
                    continue
                x = pl.Series(target, [val])
                logp = engine.logp(x, given=given)

                given[target] = val

                ixs.append(ix)
                ps.append(exp(-logp))

            return ixs, np.array(ps)
        case "unc":
            assert data is not None
            return predopt_unc(engine, target, data)
        case "abserr" | "relerr" | "sqerr":
            assert data is not None
            return predopt_err(engine, target, data, objective)
        case _:
            raise ValueError(f"Unsupported obejctive `{objective}`")
