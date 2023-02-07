import itertools as it
import numpy as np
import polars as pl
import plotly.express as px

import pylace_core

from pylace import utils


class ClusterMap:
    def __init__(self, df: pl.DataFrame, linkage: np.ndarray, figure=None):
        self.df = df
        self.figure = figure
        self.linkage = linkage


class Engine:
    def __init__(self, *args, **kwargs):
        '''Create a new engine

        Create a new engine by loading metadata or by supplying a dataset and
        drawing from the prior.

        Parameters
        ----------
        metadata: path-like, optional
            Path to metadata.
        data_source: path-like, optional
            The path to the dataset. Can be csv, csv.gz, feather (IPC), parquet,
            json, and json lines. The type will be inferred from the file
            extension unless ``source_type`` is provided explicitly. Note that
            ``data_source`` is required if ``metadata`` is not provded.
        codebook: path-like, optional
            The path the codebook. If ``None`` (default) a default codebook will
            be generated.
        n_states: int, optional
            The number of states to initialize. The default it 16.
        id_offset: int, optional
            A number to add to the state IDs. Used to facilitate easily merging
            multiple metadata files that were run on different machines. By
            default no offset (0) is provided.
        rng_seed: int, optional
            Seed for the random number generator. If ``None`` (default), the RNG
            is seeded by the system.
        source_type: str, optional
            The type of ``data_source``. Can be 'csv', 'csv.gz', 'feather',
            'ipc', 'parquet', 'json', or 'jsonl'. If ``None`` (default) the type
            is inferred from the ``data_source`` file extension.
        cat_cutoff: int, optional
            The max integer value a column can take on before it modeled as
            Count type rather than Categorial type. Only used if ``codebook`` is
            ``None``. Default is 20.
        no_hypers: bool, optional
            Only used if ``codebook`` is ``None``. If ``True``, hyper prior
            will be disables during default codebook creation. The default is
            ``False``.

        Returns
        -------
        Engine
            The engine

        Examples
        --------

        Load an Engine

        >>> from pylace import Engine
        >>> Engine(metadata='metadata.rp')

        Create an Engine from the prior

        >>> Engine(data='data.csv', codebook='codebook.yaml')
        '''
        if "metadata" in kwargs:
            if len(kwargs) > 1:
                raise ValueError('No other arguments may be supplied with \
                                 the `metadata` argument')
            engine = pylace_core.Engine.load(kwargs['metadata'])
        else:
            engine = pylace_core.Engine(*args, **kwargs)
        self.engine = engine

    @property
    def index(self) -> list:
        '''The rows IDs
        '''
        return self.engine.index

    @property
    def column(self) -> list[str]:
        '''The column names
        '''
        return self.engine.index

    @property
    def n_states(self) -> int:
        '''The number of states
        '''
        return self.engine.n_states

    @property
    def n_rows(self) -> int:
        '''The number of rows
        '''
        return self.engine.n_rows

    @property
    def n_cols(self) -> int:
        '''The number of columns
        '''
        return self.engine.n_cols

    @property
    def shape(self) -> tuple[int, int]:
        '''A (n_rows, n_cols) tuple
        '''
        return self.engine.shape

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
        >>> from pylace.examples import Satellites
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
            return self.engine.logp_scaled(values, given, col_max_logps)
        else:
            return self.engine.logp(values, given)

    def surprisal(self, col, *, rows=None, values=None, state_ixs=None):
        return self.engine.surprisal(col, rows, values, state_ixs)

    # Compute the inconsistency of the values
    def inconsistency(self, values, given=None):
        logps = self.engine.logp(values, given=given)
        if given is None:
            pass
            marg = sum([self.engine.logp(values[col]) for col in values.columns])
        else:
            marg = self.engine.logp(values)

        out = logps / marg

        if isinstance(out, polars.Series):
            out.name = 'inconsistency'

        return out

    def novelty(self, row, *, wrt=None):
        return self.engine.novelty(row, wrt)

    def entropy(self, cols, n_mc_samples=1000):
        return self.engine.entropy(cols, n_mc_samples)

    def predict(self, target, given=None, *, with_uncertainty=True):
        return self.engine.predict(target, given, with_uncertainty)

    def impute(self, col, rows=None, *, unc_type='js_divergence'):
        return self.engine.impute(col, rows, unc_type)

    def rowsim(self, row_pairs, *, wrt=None, col_weighted=False):
        return self.engine.rowsim(row_pairs, wrt, col_weighted)

    def depprob(self, col_pairs):
        return self.engine.depprob(col_pairs)

    def mi(self, col_pairs, n_mc_samples=1000, *, mi_type='Iqr'):
        return self.engine.mi(col_pairs, n_mc_samples, mi_type)

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
