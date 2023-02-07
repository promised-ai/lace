import itertools as it
import numpy as np
import polars as pl
import plotly.express as px

import lace_core

from lace import utils


class ClusterMap:
    def __init__(self, df: pl.DataFrame, linkage: np.ndarray, figure=None):
        self.df = df
        self.figure = figure
        self.linkage = linkage


class Engine(lace_core.Engine):
    def logp(self, values, given=None, *, scaled=False, col_max_logps=None):
        '''Compute the log likelihood

        This function computes ``log p(values)`` or ``log p(values|given)``.

        Parameters
        ----------
        values: polars or pandas DataFrame or Series
            The values over which to compute the log likelihood. Each row of the
            DataFrame, or each entry of the Series, is an observation. Column
            names (or the Series name) should correspond to names of features in
            the table.
        given: dict[index, value], optional
            A dictionary mapping column indices/name to values, which specifies
            conditions on the observations.
        scaled: bool, optional
            If `True` the components of the likelihoods will be scaled so that
            each dimension (feature) contributes a likelihood in [0, 1], thus
            the scaled log likelihood will not be as prone to being dominated
            by any one feature.
        col_max_logps: list[float], optional
            The cache used for scaling.

        Returns
        -------
        polars.Series or float
            The log likelihood for each observation in ``values``

        Notes
        -----
        - For missing-not-at-random (MNAR) columns, asking about the likelihood
          of a values returns the likelihood of just that value; not the
          likelihood of that value and that value being present. Computing logp
          of ``None`` returns the log likeihood of a value being missing.
        - The ``scaled`` variant is a heuristic used for model monitoring.

        Examples
        --------

        Ask about the likelihood of values in a single column
        
        >>> import polars as pl
        >>> from lace.examples import Satellites
        >>>
        >>> engine = Satellites()
        >>> class_of_orbit = pl.Series('Class_of_Orbit', ['LEO', 'MEO', 'GEO'])
        >>> engine.logp(class_of_orbit).exp()
        shape: (3,)
        Series: 'logp' [f64]
        [
            0.516332
            0.068346
            0.383582
        ]

        Conditioning using ``given``

        >>> engine.logp(class_of_orbit, given={'Period_minutes': 1436.0})
        shape: (3,)
        Series: 'logp' [f64]
        [
            0.000277
            0.000715
            0.998543
        ]

        Ask about the likelihood of values belonging to multiple features

        >>> values = pl.DataFrame({
        ...   'Class_of_Orbit': ['LEO', 'MEO', 'GEO'],
        ...   'Period_minutes': [70.0, 320.0, 1440.0],
        ... })
        >>> engine.logp(values)
        shape: (3,)
        Series: 'logp' [f64]
        [
            0.000339
            0.000011
            0.016768
        ]

        For columns which we explicitly model missing-not-at-random data, we can
        ask about the likelihood of missing values.

        >>> from math import exp
        >>> no_long_geo = pl.Series('longitude_radians_of_geo', [None])
        >>> exp(engine.logp(no_long_geo))
        0.6300749054787617

        The probability of a value missing (not-at-random) changes depending on
        the conditions.

        >>> exp(engine.logp(no_long_geo, given={'Class_of_Orbit': 'GEO'}))
        0.058265789933391016

        And we can condition on missingness

        >>> engine.logp(
        ...   class_of_orbit,
        ...   given={'longitude_radians_of_geo': None},
        ... ).exp()
        shape: (3,)
        Series: 'logp' [f64]
        [
            0.816206
            0.102968
            0.035501
        ]
        '''
        if scaled:
            # TODO: add a class method to compute the cache
            return self.logp_scaled(values, given, col_max_logps)
        else:
            return self.logp(values, given)

    # Compute the inconsistency of the values
    def inconsistency(self, values, given=None):
        logps = self.logp(values, given=given)
        if given is None:
            pass
            marg = sum([self.logp(values[col]) for col in values.columns])
        else:
            marg = self.logp(values)

        out = logps / marg

        if isinstance(out, pl.Series):
            out.name = 'inconsistency'

        return out

    def pairwise_fn(self, fn_name, indices=None, **kwargs):
        if indices is not None:
            pairs = list(it.product(indices, indices))
        else:
            pairs, _ = utils.get_all_pairs(fn_name, self.engine)

        return self.engine.pairwise_fn(fn_name, pairs, kwargs)

    # Generate a Plotly-based clustermap
    def clustermap(self, fn_name, *, indices=None, linkage_method='ward', no_plot=False, fn_kwargs={}, **kwargs):
        fn = self.pairwise_fn(fn_name, indices, **fn_kwargs)

        df = fn.pivot(values=fn_name, index='A', columns='B')
        df, linkage = utils.hcluster(df, method=linkage_method)

        if not no_plot:
            fig = px.imshow(
                df[:, 1:], 
                labels=dict(x='A', y='B', color=fn_name),
                y=df['A'],
                **kwargs
            )
            return ClusterMap(df, linkage, fig)
        else:
            return ClusterMap(df, linkage)
