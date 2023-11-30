"""Plottling utilities."""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import matplotlib.pyplot as plt

from lace import Engine


def diagnostics(
    engine: Engine,
    name: str = "score",
    log_x: bool = False,
) -> go.Figure:
    """
    Plot state diagnostics.

    Parameters
    ----------
    engine: lace.Engine
        The engine whose diagnostics to plot
    name: str, optional
        The name of the diagnostic to plot
    log_x: bool, optional
        If True, plot on a log x-axis

    Returns
    -------
    plotly.graph_objects.Figure
        The figure handle

    Examples
    --------
    Plot the score over iterations for satellites data

    >>> from lace.examples import Satellites
    >>> from lace.plot import diagnostics
    >>> diagnostics(Satellites(), log_x=True).show()
    """
    diag = engine.diagnostics(name)
    step = np.arange(diag.shape[0])

    mean = diag.mean(axis=1).rename("mean")

    df = (
        diag.with_columns(mean)
        .with_columns(pl.Series("step", step))
        .melt(
            value_vars=[c for c in diag.columns if name in c],
            id_vars=["step"],
            value_name=name,
            variable_name="state",
        )
    )

    title = f"{name} over iterations"

    fig = px.line(
        df.to_pandas(),
        x="step",
        y=name,
        title=title,
        color="state",
        log_x=log_x,
    )

    fig.add_trace(
        go.Scatter(
            x=step,
            y=mean,
            mode="lines",
            name="mean",
            line={"color": "black", "width": 5},
            connectgaps=True,
        )
    )
    return fig


def _predict_xs(
    engine, target, given, *, n_points=1_000, range_stds=2.5
) -> pl.Series:
    ftype = engine.ftype(target)
    if ftype == "Continuous":
        xs = engine.simulate([target], given=given, n=10_000)
        mean = xs[target].mean()
        std = xs[target].std()
        width = range_stds * std
        xs = np.linspace(mean - width, mean + width, n_points)
        return pl.Series(target, xs)
    elif ftype == "Categorical":
        return pl.Series(target, engine.engine.categorical_support(target))
    else:
        raise ValueError("unsupported ftype")


def prediction_uncertainty(
    engine: Engine,
    target: Union[str, int],
    given: Optional[Dict[Union[str, int], object]] = None,
    xs: Optional[Union[pl.Series, pd.Series]] = None,
    n_points: int = 1_000,
    range_stds: float = 3.0,
):
    """
    Visualize prediction uncertainty.

    Parameters
    ----------
    engine: Engine
        The Engine from which to predict
    target: column index
        The column to predict
    given: Dict[column index, value], optional
        Column -> Value dictionary describing observations. Note that
        columns can either be indices (int) or names (str)
    xs: polars.Series or pandas.Series, optional
        The values over which to visualize uncertainty. If None (default),
        values will be computed manually. For categorical columns, the value
        map will be used; for continuous and count columns the values +/-
        ``range_stds`` standard deviations from the mean will be used.

    Examples
    --------
    Visualize uncertainty for a continuous target

    >>> from lace.examples import Satellites
    >>> from lace.plot import prediction_uncertainty
    >>> satellites = Satellites()
    >>> fig = prediction_uncertainty(
    ...     satellites,
    ...     "Period_minutes",
    ...     given={"Class_of_Orbit": "GEO"},
    ... )
    >>> fig.show()

    Narrow down the range for visualization

    >>> import numpy as np
    >>> import polars as pl
    >>> fig = prediction_uncertainty(
    ...     satellites,
    ...     "Period_minutes",
    ...     given={"Class_of_Orbit": "GEO"},
    ...     xs=pl.Series("Period_minutes", np.linspace(1350, 1500, 500)),
    ... )
    >>> fig.show()

    Visualize uncertainty for a categorical target

    >>> fig = prediction_uncertainty(
    ...     satellites,
    ...     "Class_of_Orbit",
    ...     given={"Period_minutes": 1326.0},
    ... )
    >>> fig.show()
    """
    pred, unc = engine.predict(target, given=given)

    n_states = engine.n_states

    if xs is None:
        xs = _predict_xs(
            engine, target, given, n_points=n_points, range_stds=range_stds
        )

    title = f"{target} uncertainty: {unc}"

    fig = px.line(title=title).update_layout(
        xaxis_title=target, yaxis_title="Likelihood"
    )

    for state_ix in range(n_states):
        ys = engine.logp(xs, given, state_ixs=[state_ix]).exp()
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"State {state_ix}",
                line={"color": "rgba(150, 150, 150, 0.5)", "width": 1},
                connectgaps=True,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=engine.logp(xs, given).exp(),
            mode="lines",
            name="Mean",
            line={"color": "#333333", "width": 4},
            connectgaps=True,
        )
    )

    fig.add_vline(
        x=pred,
        line_width=1.5,
        line_dash="dash",
        line_color="red",
        name="Prediction",
    )

    return fig


def _get_xs(engine, cat_rows, view_cols, compute_ps=False):
    xs = engine[cat_rows, view_cols]
    if isinstance(xs, (int, float)):
        xs = np.array(xs)
        if compute_ps:
            ps = [0.0]
    else:
        xs = xs[:, 1:]
        if compute_ps:
            ps = engine.logp(xs)
        xs = xs.to_numpy()

    if compute_ps:
        return xs, ps
    else:
        return xs

def state(
    engine: Engine,
    state_ix: int,
    missing_color = None,
    cat_gap:int = 1,
    view_gap:int = 1,
    ax = None,
):
    if ax is None:
        ax = plt.gca()

    n_rows, n_cols = engine.shape
    col_asgn = engine.column_assignment(state_ix)
    row_asgns = engine.row_assignments(state_ix)

    n_views = len(row_asgns)
    max_cats = max(max(asgn) + 1 for asgn in row_asgns)

    dim_col = n_cols + (n_views - 1) * view_gap
    dim_row = n_rows + (max_cats - 1) * cat_gap

    zs = np.zeros((dim_row, dim_col)) + 0.1

    row_names = engine.index
    col_names = engine.columns 

    col_start = 0
    for view_ix, row_asgn in enumerate(row_asgns):
        row_start = 0
        n_cats = max(row_asgn) + 1
        view_len = sum(z == view_ix for z in col_asgn)
        view_cols = [i for i, z in enumerate(col_asgn) if z == view_ix]

        # sort columns within each view
        ps = []
        for col in view_cols:
            ps.append(engine.logp(engine[col][:, 1:]).sum())
        ixs = np.argsort(ps)[::-1]
        view_cols = [view_cols[ix] for ix in ixs]

        for i, col in enumerate(view_cols):
            ax.text(col_start + i, -1, col_names[col], ha='center', va='bottom', rotation=90)

        for cat_ix in range(n_cats):
            cat_len = sum(z == cat_ix for z in row_asgn)
            cat_rows = [i for i, z in enumerate(row_asgn) if z == cat_ix]

            xs, ps = _get_xs(engine, cat_rows, view_cols, compute_ps=True)
            ixs = np.argsort(ps)[::-1]
            cat_rows = [cat_rows[ix] for ix in ixs]
            xs = _get_xs(engine, cat_rows, view_cols)
            zs[row_start:row_start+cat_len, col_start:col_start + view_len] = xs

            # label rows
            for i, row in enumerate(cat_rows):
                ax.text(col_start-1, i + row_start, row_names[row], ha='right', va='center')

            row_start += cat_len + cat_gap

        col_start += view_len + view_gap


    ax.matshow(zs, cmap='gray_r')



if __name__ == "__main__":
    from lace.examples import Animals

    eng = Animals()
    plt.figure(tight_layout=True, facecolor='#e8e8e8')
    ax = plt.gca()
    state(eng, 1, view_gap=15, ax=ax)
    plt.axis('off')
    plt.show()
    # import doctest

    # doctest.testmod()
