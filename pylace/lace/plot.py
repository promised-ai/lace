"""
Plottling utilities
"""
import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go

from lace import Engine


def diagnostics(
    engine: Engine,
    name: str = "score",
    log_x: bool = False,
) -> go.Figure:
    """
    Plot state diagnostics

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

    mean = diag.mean(axis=1).rename(f"mean")

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
            line=dict(color="black", width=5),
            connectgaps=True,
        )
    )
    return fig


if __name__ == "__main__":
    import doctest

    doctest.testmod()
