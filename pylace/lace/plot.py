"""Plotting utilities"""

from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

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

    if xs is None:
        if compute_ps:
            return np.array([None]), [0.0]
        else:
            return np.array([None])

    if isinstance(xs, (int, float)):
        xs = np.array(xs)
        if compute_ps:
            ps = [0.0]
    else:
        xs = xs[:, 1:]

        if compute_ps:
            n_rows, n_cols = xs.shape
            ps = []
            for row_ix in range(n_rows):
                row = xs[row_ix, :]
                null_count = row.null_count()
                to_drop = [c for c in row.columns if null_count[0, c] > 0]
                k = n_cols - len(to_drop)
                if k == 0:
                    ps.append(float("-inf"))
                    continue

                row = row.drop(to_drop)
                p = engine.logp(row) / k
                ps.append(p)

        xs = xs.to_numpy()

    if compute_ps:
        return xs, ps
    else:
        return xs


def _makenorm(cmap, missing_color, *, mapper=None, xlim=None):
    if mapper is not None:

        def _fn(val):
            k = len(mapper) - 1
            return mapper[val] / k

    else:
        xmin, xmax = xlim

        def _fn(val):
            return (val - xmin) / (xmax - xmin)

    def _norm(val):
        colormap = plt.cm._colormaps[cmap]
        if pd.isnull(val):
            return missing_color
        else:
            return np.array(colormap(_fn(val)))

    return _norm


def _get_colors(engine: Engine, *, cmap: str = "gray_r", missing_color=None):
    codebook = engine.codebook

    if missing_color is None:
        missing_color = np.array([1.0, 0.0, 0.2, 1.0])

    n_rows, n_cols = engine.shape
    colors = np.zeros((n_rows, n_cols, 4))

    for i, col in enumerate(engine.columns):
        ftype = engine.ftype(col)
        xs = engine[col][col]
        if ftype == "Categorical":
            valmap = codebook.value_map(col)
            mapper = {v: k for k, v in enumerate(valmap.values())}

            _norm = _makenorm(cmap, missing_color, mapper=mapper)
        else:
            _norm = _makenorm(cmap, missing_color, xlim=(xs.min(), xs.max()))

        colors[:, i] = np.array([_norm(x) for x in xs])

    return colors


def state(
    engine: Engine,
    state_ix: int,
    *,
    cmap: Optional[str] = None,
    missing_color=None,
    cat_gap: int = 1,
    view_gap: int = 1,
    show_index: bool = True,
    show_columns: bool = True,
    min_height: int = 0,
    min_width: int = 0,
    aspect=None,
    ax=None,
):
    """
    Plot a Lace state.

    View are sorted from largest (most columns) to smallest. Within views,
    columns are sorted from highest (left) to lowest total likelihood.
    Categories are sorted from largest (most rows) to smallest. Within
    categories, rows are sorted from highest (top) to lowest log likelihood.

    Parameters
    ----------
    engine: Engine
        The engine containing the states to plot
    state_ix: int
        The index of the state to plot
    cmap: str, optional, default: gray_r
        The color map to use for present data
    missing_color: optional, default: red
        The RGBA array representation ([float, float, float, float]) of the
        color to use to represent missing data
    cat_gap: int, optional, default: 1
        The vertical spacing (in cells) between categories
    view_gap: int, optional, default: 1
        The horizontal spacing (in cell) between views
    show_index: bool, default: True
        If True (default), will show row names next to rows in each view
    show_columns: bool, default: True
        If True (default), will show columns names above each column
    min_height: int, default: 0
        The minimum height in cells of the state render. Padding will be added
        to the lower part of the image.
    min_width: int (default: 0)
        The minimum width in cells of the state render. Padding will be added
        to the right of the image.
    aspect: {'equal', 'auto'} or float or None, default: None
        matplotlib imshow aspect
    ax: matplotlib.Axis, optional
        The axis on which to plot

    Examples
    --------

    Render an animals state

    >>> import matplotlib.pyplot as plt
    >>> from lace.examples import Animals
    >>> from lace import plot
    >>> engine = Animals()
    >>> fig = plt.figure(tight_layout=True, facecolor="#00000000")
    >>> ax = plt.gca()
    >>> plot.state(
    ...    engine,
    ...    7,
    ...    view_gap=13,
    ...    cat_gap=3,
    ...    ax=ax,
    ... )
    >>> _ = plt.axis("off")
    >>> plt.show()


    Render a satellites State, which has continuous, categorial and
    missing data

    >>> from lace.examples import Satellites
    >>> engine = Satellites()
    >>> fig = plt.figure(tight_layout=True, facecolor="#00000000")
    >>> ax = plt.gca()
    >>> plot.state(
    ...    engine,
    ...    1,
    ...    view_gap=2,
    ...    cat_gap=100,
    ...    show_index=False,
    ...    show_columns=False,
    ...    ax=ax,
    ...    cmap="YlGnBu",
    ...    aspect="auto"
    ... )
    >>> _ = plt.axis("off")
    >>> plt.show()
    """
    if ax is None:
        ax = plt.gca()

    if cmap is None:
        cmap = "gray_r"

    n_rows, n_cols = engine.shape
    col_asgn = engine.column_assignment(state_ix)
    row_asgns = engine.row_assignments(state_ix)

    n_views = len(row_asgns)
    max_cats = max(max(asgn) + 1 for asgn in row_asgns)

    dim_col = n_cols + (n_views - 1) * view_gap
    dim_row = n_rows + (max_cats - 1) * cat_gap

    zs = np.zeros((dim_row, dim_col, 4))

    row_names = engine.index
    col_names = engine.columns

    view_counts = np.bincount(col_asgn)
    view_ixs = np.argsort(view_counts)[::-1]

    colors = _get_colors(engine, cmap=cmap, missing_color=missing_color)

    col_start = 0
    for view_ix in view_ixs:
        row_asgn = row_asgns[view_ix]
        row_start = 0
        view_cols = [i for i, z in enumerate(col_asgn) if z == view_ix]
        view_len = len(view_cols)

        # sort columns within each view
        ps = []
        for col in view_cols:
            xs = engine[col][:, 1:].drop_nulls()
            if xs.shape[0] == 0:
                ps.append(float("-inf"))
            ps.append(engine.logp(xs).sum() / xs.shape[0])
        ixs = np.argsort(ps)[::-1]
        view_cols = [view_cols[ix] for ix in ixs]

        if show_columns:
            for i, col in enumerate(view_cols):
                ax.text(
                    col_start + i,
                    -1,
                    col_names[col],
                    ha="center",
                    va="bottom",
                    rotation=90,
                )

        max(row_asgn) + 1
        cat_counts = np.bincount(row_asgn)
        cat_ixs = np.argsort(cat_counts)[::-1]

        for cat_ix in cat_ixs:
            cat_rows = [i for i, z in enumerate(row_asgn) if z == cat_ix]
            cat_len = len(cat_rows)

            xs, ps = _get_xs(engine, cat_rows, view_cols, compute_ps=True)
            ixs = np.argsort(ps)[::-1]
            cat_rows = [cat_rows[ix] for ix in ixs]

            cs = np.zeros((len(cat_rows), len(view_cols), 4))
            for iix, i in enumerate(cat_rows):
                for jix, j in enumerate(view_cols):
                    cs[iix, jix] = colors[i, j]
            # cs = colors[cat_rows, view_cols, :]
            zs[
                row_start : row_start + cat_len,
                col_start : col_start + view_len,
            ] = cs

            # label rows
            if show_index:
                for i, row in enumerate(cat_rows):
                    ax.text(
                        col_start - 1,
                        i + row_start,
                        row_names[row],
                        ha="right",
                        va="center",
                    )

            ax.text(
                col_start + view_counts[view_ix] / 2.0 - 0.5,
                row_start + cat_counts[cat_ix] + cat_gap * 0.15,
                f"$C_{{{cat_ix}}}$",
                ha="center",
                va="center",
            )

            row_start += cat_len + cat_gap

        ax.text(
            col_start + view_counts[view_ix] / 2.0 - 0.5,
            dim_row + cat_gap,
            f"$V_{{{view_ix}}}$",
            ha="left",
            va="top",
        )
        col_start += view_len + view_gap

    if min_height > zs.shape[0]:
        margin = min_height - zs.shape[0]
        zs = np.vstack((zs, np.zeros((margin, zs.shape[1], 4))))

    if min_width > zs.shape[1]:
        margin = min_width - zs.shape[1]
        zs = np.hstack((zs, np.zeros((zs.shape[0], margin, 4))))

    ax.matshow(zs, aspect=aspect)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
