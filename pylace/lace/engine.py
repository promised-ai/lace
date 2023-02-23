"""
The main interface to Lace models
"""
from os import PathLike
import itertools as it
from typing import Union, Optional
import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import lace_core

from lace import utils


class ClusterMap:
    """
    Contains information about a pairwise function computed over a number of
    values.

    Attributes
    ----------
    df: polars.DataFrame
        The function data. Each column is named after a value. There is an
        'index' column that contains the other value in the pair.
    linkage: numpy.ndarray
        scipy linkage computed during hierarchical clustering
    figure: plotly.Figure, optional
        The handle to the plotly heatmap figure. May not be included if the
        user chose not to plot the ``clustermap``.
    """

    def __init__(self, df: pl.DataFrame, linkage: np.ndarray, figure=None):
        self.df = df
        self.figure = figure
        self.linkage = linkage


class Engine:
    def __init__(self, *args, **kwargs):
        """
        Load or create a new ``Engine``

        Parameters
        ----------
        metadata: path-like, optional
            The path to the metadata to load. If ``metadata`` is provided, no
            other arguments may be provided.
        data_source: path-like, optional
            The path to the source data file.
        codebook: path-like, optional
            Path to the codebook. If ``None`` (default), a codebook is inferred.
        n_states: usize
            The number of states (independent Markov chains). default is 16.
        id_offset: int
            An offset for renaming states in the metadata. Used when training a
            single engine on multiple machines. If one wished to split an
            8-state ``Engine`` run on to two machine, one may run a 4-state
            ``Engine`` on the first machine, then a 4-state ``Engine`` on the
            second machine with ``id_offset=4``. The states within two metadata
            files may be merged by copying without name collisions.
        rng_seed: int
            Random number generator seed
        source_type: str, optional
            The type of the source file. If ``None`` (default) the type is
            inferred from the file extension.
        cat_cutoff: int, optional
            The maximum integer value an all-integer column takes on at which
            it is considered count type.
        no_hypers: bool
            If ``True``, hyper priors, and prior parameter inference will be
            disabled


        Examples
        --------

        Load an Engine from metadata

        >>> from lace import Engine                   # doctest: +SKIP
        >>> engine = Engine(metadata='metadata.lace') # doctest: +SKIP

        Create a new Engine with default codebook. The start state is drawn from
        the probabilistic cross-categorization prior.

        >>> engine = Engine(data_source='data.csv', n_states=32) # doctest: +SKIP
        """
        if "metadata" in kwargs:
            if len(kwargs) > 1:
                raise ValueError(
                    "No other arguments may be privded if \
                                 `metadata` is provided"
                )
            self.engine = lace_core.CoreEngine.load(kwargs["metadata"])
        else:
            self.engine = lace_core.CoreEngine(*args, **kwargs)

    def save(self, path: Union[str, bytes, PathLike]):
        """
        Save the Engine metadata to ``path``

        Examples
        --------

        Save a copy of an engine

        >>> from lace import Engine                    # doctest: +SKIP
        >>> engine = Engine(metadata='metadata.lace')  # doctest: +SKIP
        >>> engine.save('metadata-copy.lace')          # doctest: +SKIP
        """
        self.engine.save(path)

    @property
    def shape(self):
        """
        A tuple containing the number of rows and the number of columns in the table

        Examples
        --------
        >>> from lace.examples import Satellites
        >>> engine = Satellites()
        >>> engine.shape
        (1164, 20)
        """
        return self.engine.shape

    @property
    def n_rows(self):
        """
        The number of rows in the table

        Examples
        --------
        >>> from lace.examples import Satellites
        >>> engine = Satellites()
        >>> engine.n_rows
        1164
        """
        return self.engine.n_rows

    @property
    def n_cols(self):
        """
        The number of columns in the table

        Examples
        --------
        >>> from lace.examples import Satellites
        >>> engine = Satellites()
        >>> engine.n_cols
        20
        """
        return self.engine.n_cols

    @property
    def n_states(self):
        """
        The number of states (independent Markov chains)

        Examples
        --------
        >>> from lace.examples import Satellites
        >>> engine = Satellites()
        >>> engine.n_states
        16
        """
        return self.engine.n_states

    @property
    def columns(self):
        """
        A list of the column names appearing in their order in the table

        Examples
        --------
        >>> from lace.examples import Satellites
        >>> engine = Satellites()
        >>> engine.columns  # doctest: +NORMALIZE_WHITESPACE
        ['Country_of_Operator',
         'Users',
         'Purpose',
         'Class_of_Orbit',
         'Type_of_Orbit',
         'Perigee_km',
         'Apogee_km',
         'Eccentricity',
         'Period_minutes',
         'Launch_Mass_kg',
         'Dry_Mass_kg',
         'Power_watts',
         'Date_of_Launch',
         'Expected_Lifetime',
         'Country_of_Contractor',
         'Launch_Site',
         'Launch_Vehicle',
         'Source_Used_for_Orbital_Data',
         'longitude_radians_of_geo',
         'Inclination_radians']
        """
        return self.engine.columns

    @property
    def index(self):
        """
        The string row names of the engine
        """
        return self.engine.index

    @property
    def ftypes(self):
        """
        A dictionary mapping column names to feature types

        Examples
        --------
        >>> from lace.examples import Satellites  # doctest: +SKIP
        >>> engine = Satellites()                 # doctest: +SKIP
        >>> engine.ftypes                         # doctest: +SKIP
        {'Date_of_Launch': 'Continuous',
         'Purpose': 'Categorical',
         'Period_minutes': 'Continuous',
         'Expected_Lifetime': 'Continuous',
         'longitude_radians_of_geo': 'Continuous',
         'Inclination_radians': 'Continuous',
         'Apogee_km': 'Continuous',
         'Country_of_Contractor': 'Categorical',
         'Eccentricity': 'Continuous',
         'Source_Used_for_Orbital_Data': 'Categorical',
         'Perigee_km': 'Continuous',
         'Dry_Mass_kg': 'Continuous',
         'Country_of_Operator': 'Categorical',
         'Power_watts': 'Continuous',
         'Launch_Site': 'Categorical',
         'Launch_Vehicle': 'Categorical',
         'Type_of_Orbit': 'Categorical',
         'Users': 'Categorical',
         'Launch_Mass_kg': 'Continuous',
         'Class_of_Orbit': 'Categorical'}
        """
        return self.engine.ftypes

    def ftype(self, col: str | int):
        """
        Get the feature type of a column

        Parameters
        ----------
        col: column index
            The column index

        Returns
        -------
        str
            The feature type

        Examples
        --------
        >>> from lace.examples import Satellites
        >>> engine = Satellites()
        >>> engine.ftype('Class_of_Orbit')
        'Categorical'
        >>> engine.ftype('Period_minutes')
        'Continuous'
        """
        return self.engine.ftype(col)

    def column_assignment(self, state_ix: int) -> list[int]:
        """
        Return the assignment of columns to views

        Parameters
        ----------
        state_ix: int
            The state index for which to pull the column assignment

        Returns
        -------
        asgn: list[int]
            `asgn[i]` is the index of the view to which column i is assigned

        Examples
        --------

        Get the assignment of columns to views in state 0

        >>> from lace.examples import Satellites
        >>> engine = Satellites()
        >>> engine.column_assignment(0)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        """
        return self.engine.column_assignment(state_ix)

    def row_assignments(self, state_ix: int):
        """
        Return the assignment of rows to categories for each view

        Parameters
        ----------
        state_ix: int
            The state index for which to pull the column assignment

        Returns
        -------
        asgn: list[list[int]]
            `asgn[j][i]` is the index of the category to which row i is assigned
            under view j.

        Examples
        --------

        Get the assignment category index of row 11 in view 1 of state 0

        >>> from lace.examples import Animals
        >>> animals = Animals()
        >>> animals.row_assignments(0)[1][11]
        3
        """
        return self.engine.row_assignments(state_ix)

    def __getitem__(self, ix: str | int):
        return self.engine[ix]

    def diagnostics(self, name: str = "score"):
        """Get convergence diagnostics

        Parameters
        ----------
        name: str
            'loglike', 'logprior', or 'score' (default).

        Returns
        -------
        polars.DataFrame
            Contains a column for the diagnostic for each state. Each row
            corresponds to an iteration  of the Markov chain.

        Examples
        --------

        Get the state scores

        >>> from lace.examples import Animals
        >>> animals = Animals()
        >>> diag = animals.diagnostics()
        >>> diag.shape
        (5000, 16)
        >>> diag[:, :4]  # doctest: +NORMALIZE_WHITESPACE
        shape: (5000, 4)
        ┌──────────────┬──────────────┬──────────────┬──────────────┐
        │ score_0      ┆ score_1      ┆ score_2      ┆ score_3      │
        │ ---          ┆ ---          ┆ ---          ┆ ---          │
        │ f64          ┆ f64          ┆ f64          ┆ f64          │
        ╞══════════════╪══════════════╪══════════════╪══════════════╡
        │ -2882.424453 ┆ -2809.0876   ┆ -2638.714156 ┆ -2604.137622 │
        │ -2695.299327 ┆ -2666.497867 ┆ -2608.185358 ┆ -2576.545684 │
        │ -2642.539971 ┆ -2532.638368 ┆ -2576.463401 ┆ -2568.516617 │
        │ -2488.369418 ┆ -2513.134161 ┆ -2549.299382 ┆ -2554.131179 │
        │ ...          ┆ ...          ┆ ...          ┆ ...          │
        │ -1972.005746 ┆ -2122.788121 ┆ -1965.921104 ┆ -1969.328651 │
        │ -1966.516529 ┆ -2117.398333 ┆ -1993.351756 ┆ -1986.589833 │
        │ -1969.400394 ┆ -2147.941128 ┆ -1968.697139 ┆ -1988.805311 │
        │ -1920.217666 ┆ -2081.368421 ┆ -1909.655836 ┆ -1920.432849 │
        └──────────────┴──────────────┴──────────────┴──────────────┘
        """
        df = pl.DataFrame()

        diag = self.engine.diagnostics()

        for ix in range(self.n_states):
            srs = utils._diagnostic(name, diag[ix])
            df = df.with_columns(srs.rename(f"{name}_{ix}"))

        return df

    def append_rows(
        self,
        rows: pd.Series
        | pd.DataFrame
        | pl.DataFrame
        | dict[str, dict[str, object]],
    ):
        """
        Append new rows to the table

        Parameters
        ----------
        rows: polars.DataFrame, pandas.DataFrame, pandas.Series, dict[str, dict]
            The rows to append to the table. When using a DataFrame, the index
            indicates the row names. When using a polars DataFrame, an `index`
            column must be explicitly provided. When using a pandas Series, the
            index corresponds to the feature names and the Series name
            corresponds to the row name. When using a dict, the outer dict maps
            string row names to dictionaries that map string feature names to
            values. See examples below.

        Examples
        --------

        You can append new rows as a `polars.DataFrame`. Note that the index
        must be explicitly added.

        >>> import polars as pl
        >>> from lace.examples import Animals
        >>> engine = Animals()
        >>> crab_and_sponge = pl.DataFrame({
        ...   'index': ['crab', 'sponge'],
        ...   'water': [1, 1],
        ...   'flippers': [0, 0],
        ... })
        >>> engine.append_rows(crab_and_sponge)
        >>> engine.index[-1]
        'sponge'
        >>> engine['water'][-1]
        1

        You can append new rows as a `pandas.DataFrame`,

        >>> import pandas as pd
        >>> engine = Animals()
        >>> crab_and_sponge = pd.DataFrame({
        ...   'index': ['crab', 'sponge'],
        ...   'water': [1, 1],
        ...   'flippers': [0, 0],
        ... }).set_index('index')
        >>> engine.append_rows(crab_and_sponge)
        >>> engine.index[-1]
        'sponge'
        >>> engine['water'][-1]
        1

        or a `pandas.Series`

        >>> squid = pd.Series([0, 1], index=['water', 'slow'], name='squid')
        >>> engine.append_rows(squid)
        >>> engine.index[-1]
        'squid'
        >>> engine['slow'][-1]
        1

        or a dictionary of dictionaries

        >>> engine = Animals()
        >>> rows = {
        ...   'crab': { 'water': 1, 'flippers': 0},
        ...   'sponge': { 'water': 1, 'flippers': 0},
        ...   'squid': { 'water': 1, 'slow': 1},
        ... }
        >>> engine.append_rows(rows)
        >>> engine.index[-3:]
        ['crab', 'sponge', 'squid']
        >>> engine['flippers'][-3:]  # doctest: +NORMALIZE_WHITESPACE
        shape: (3,)
        Series: 'flippers' [u32]
        [
            0
            0
            null
        ]
        """
        if isinstance(rows, dict):
            for name, values in rows.items():
                row = pd.Series(values, name=name)
                self.engine.append_rows(row)
        else:
            self.engine.append_rows(rows)

    def update(
        self,
        n_iters: int,
        *,
        timeout: Optional[int] = None,
        checkpoint: Optional[int] = None,
        transitions: Optional[lace_core.StateTransition] = None,
        save_path: Optional[Union[str, bytes, PathLike]] = None,
        quiet: bool = False,
    ):
        """
        Update the Engine by advancing the Markov chains

        Parameters
        ----------
        n_iters: int
            The number of iterations, or steps, to advance each chain (state)
        timeout: int, optional
            The timeout in seconds, which is the maximum number of seconds any
            state should run. Note that if you have fewer cores than states
            (which is usually how it goes), then the update will run for longer
            than the timeout because not all the states will be able to run at
            the same time. If timeout is `None` (default), the run will stop
            when all requested iterations have been completed.
        checkpoint: int, optional
            The number of iterations between saves. If `save_path` is not
            supplied checkpoints do nothing.
        transitions: list[StateTransition], optional
            List of state transitions to perform. If `None` (default) a default
            set is chosen.
        save_path: pathlike, optional
            Where to save the metadata. If `None` (default) the engine is not
            saved. If `checkpoint` is provided, the `Engine` will be saved at
            checkpoints and at the end of the run. If `checkpoint` is not
            provided, the `Engine` will save only at the end of the run.

        Examples
        --------

        Simple update for 100 iterations

        >>> from lace.examples import Animals
        >>> engine = Animals()
        >>> engine.update(100)

        Perform only specific transitions and set a timeout of 30 seconds

        >>> from lace import RowKernel, StateTransition
        >>> engine.update(
        ...   100,
        ...   timeout=30,
        ...   transitions=[
        ...     StateTransition.row_assignment(RowKernel.slice()),
        ...     StateTransition.view_alphas(),
        ...   ]
        ... )
        """
        return self.engine.update(
            n_iters,
            timeout=timeout,
            checkpoint=checkpoint,
            transitions=transitions,
            save_path=save_path,
            quiet=quiet,
        )

    def entropy(self, cols, n_mc_samples: int = 1000):
        """
        Estimate the entropy or joint entropy of one or more features

        Prameters
        ---------
        col: column indices
            The columns for which to compute entropy
        n_mc_samples: int
            The number of samples to use for Monte Carlo integration in cases
            that Monte Carlo integration is used


        Returns
        -------
        h: float
            The entropy, H(cols).

        Notes
        -----
        - Entropy behaves differently for continuous variables. Continuous, or
          *differential* entropy can be negative. The same holds true for joint
          entropies with one or more continuous feature.

        Examples
        --------

        Single feature entropy

        >>> from lace.examples import Animals
        >>> animals = Animals()
        >>> animals.entropy(['slow'])
        0.6755931727528786
        >>> animals.entropy(['water'])
        0.49836129824622094

        Joint entropy

        >>> animals.entropy(['swims', 'fast'])
        0.9552642751735604

        We can use entropies to compute mutual information, I(X, Y) = H(X) +
        H(Y) - H(X, Y).

        For example, there is not a lot of shared information between whether an
        animals swims and whether it is fast. These features are not predictive
        of each other.

        >>> h_swims = animals.entropy(['swims'])
        >>> h_fast = animals.entropy(['fast'])
        >>> h_swims_and_fast = animals.entropy(['swims', 'fast'])
        >>> h_swims + h_fast - h_swims_and_fast
        3.510013543328583e-05

        But swimming and having flippers are mutually predictive, so we should
        see more mutual information.

        >>> h_flippers = animals.entropy(['flippers'])
        >>> h_swims_and_flippers = animals.entropy(['swims', 'flippers'])
        >>> h_swims + h_flippers - h_swims_and_flippers
        0.19361180218629537
        """
        return self.engine.entropy(cols, n_mc_samples)

    def logp(
        self, values, given=None, *, scaled: bool = False, col_max_logps=None
    ) -> None | float | pl.Series:
        """Compute the log likelihood

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
        >>> engine = Satellites()
        >>> class_of_orbit = pl.Series('Class_of_Orbit', ['LEO', 'MEO', 'GEO'])
        >>> engine.logp(class_of_orbit).exp()  # doctest: +NORMALIZE_WHITESPACE
        shape: (3,)
        Series: 'logp' [f64]
        [
            0.515931
            0.067117
            0.385823
        ]

        Conditioning using ``given``

        >>> engine.logp(
        ...   class_of_orbit,
        ...   given={'Period_minutes': 1436.0},
        ... ).exp()  # doctest: +NORMALIZE_WHITESPACE
        shape: (3,)
        Series: 'logp' [f64]
        [
            0.000447
            0.009838
            0.985137
        ]

        Ask about the likelihood of values belonging to multiple features

        >>> values = pl.DataFrame({
        ...   'Class_of_Orbit': ['LEO', 'MEO', 'GEO'],
        ...   'Period_minutes': [70.0, 320.0, 1440.0],
        ... })
        >>> engine.logp(values).exp()  # doctest: +NORMALIZE_WHITESPACE
        shape: (3,)
        Series: 'logp' [f64]
        [
            0.000365
            0.000018
            0.015827
        ]

        An example of the scaled variant:

        >>> from lace import ColumnMaximumLogpCache
        >>> col_max_logps = ColumnMaximumLogpCache(
        ...     engine.engine,
        ...     values.columns,
        ... )
        >>> engine.logp(
        ...     values,
        ...     col_max_logps=col_max_logps,
        ...     scaled=True,
        ... ).exp()  # doctest: +NORMALIZE_WHITESPACE
        shape: (3,)
        Series: 'logp_scaled' [f64]
        [
            0.544088
            0.056713
            2.635449
        ]

        For columns which we explicitly model missing-not-at-random data, we can
        ask about the likelihood of missing values.

        >>> from math import exp
        >>> no_long_geo = pl.Series('longitude_radians_of_geo', [None])
        >>> exp(engine.logp(no_long_geo))
        0.6269378516150409

        The probability of a value missing (not-at-random) changes depending on
        the conditions.

        >>> exp(engine.logp(no_long_geo, given={'Class_of_Orbit': 'GEO'}))
        0.06569732670635807

        And we can condition on missingness

        >>> engine.logp(
        ...   class_of_orbit,
        ...   given={'longitude_radians_of_geo': None},
        ... ).exp() # doctest: +NORMALIZE_WHITESPACE
        shape: (3,)
        Series: 'logp' [f64]
        [
            0.820026
            0.098607
            0.040467
        ]
        """
        if scaled:
            # TODO: add a class method to compute the cache
            srs = self.engine.logp_scaled(values, given, col_max_logps)
        else:
            srs = self.engine.logp(values, given)

        return utils.return_srs(srs)

    def inconsistency(self, values, given=None):
        """
        Compute inconsistency

        Parameters
        ----------
        values: polars or pandas DataFrame or Series
            The values over which to compute the inconsistency. Each row of the
            DataFrame, or each entry of the Series, is an observation. Column
            names (or the Series name) should correspond to names of features in
            the table.
        given: dict[index, value], optional
            A dictionary mapping column indices/name to values, which specifies
            conditions on the observations.

        Notes
        -----
        - If no `given` is provided, `values` must contain two or more columns.
          Since inconsistency requires context, the inconsistency of a single
          unconditioned value is always 1.

        Examples
        --------

        Compute the inconsistency of all animals over all variables

        >>> import polars as pl
        >>> from lace import examples
        >>> animals = examples.Animals()
        >>> index = animals.df['id']
        >>> inconsistency = animals.inconsistency(animals.df.drop('id'))
        >>> pl.DataFrame({
        ...     'index': index,
        ...     'inconsistency': inconsistency
        ... }).sort('inconsistency', reverse=True)  # doctest: +NORMALIZE_WHITESPACE
        shape: (50, 2)
        ┌────────────────┬───────────────┐
        │ index          ┆ inconsistency │
        │ ---            ┆ ---           │
        │ str            ┆ f64           │
        ╞════════════════╪═══════════════╡
        │ collie         ┆ 0.816517      │
        │ beaver         ┆ 0.809991      │
        │ rabbit         ┆ 0.785911      │
        │ polar+bear     ┆ 0.783775      │
        │ ...            ┆ ...           │
        │ killer+whale   ┆ 0.513013      │
        │ blue+whale     ┆ 0.503965      │
        │ dolphin        ┆ 0.480259      │
        │ humpback+whale ┆ 0.434979      │
        └────────────────┴───────────────┘

        Find satellites with inconsistent orbital periods

        >>> engine = examples.Satellites()
        >>> data = []
        >>> # examples give us special access to the underlying data
        >>> for ix, row in engine.df.to_pandas().iterrows():
        ...     given = row.dropna().to_dict()
        ...     period =  given.pop('Period_minutes', None)
        ...
        ...     if period is None:
        ...         continue
        ...
        ...     ix = given.pop('ID')
        ...     ic = engine.inconsistency(
        ...         pl.Series('Period_minutes', [period]),
        ...         given,
        ...     )
        ...
        ...     data.append({
        ...         'index': ix,
        ...         'inconsistency': ic,
        ...         'Period_minutes': period,
        ...     })
        ...
        >>> pl.DataFrame(data).sort('inconsistency', reverse=True)  # doctest: +NORMALIZE_WHITESPACE
        shape: (1162, 3)
        ┌─────────────────────────────────────┬───────────────┬────────────────┐
        │ index                               ┆ inconsistency ┆ Period_minutes │
        │ ---                                 ┆ ---           ┆ ---            │
        │ str                                 ┆ f64           ┆ f64            │
        ╞═════════════════════════════════════╪═══════════════╪════════════════╡
        │ Intelsat 903                        ┆ 1.840728      ┆ 1436.16        │
        │ QZS-1 (Quazi-Zenith Satellite Sy... ┆ 1.47515       ┆ 1436.0         │
        │ Mercury 2 (Advanced Vortex 2, US... ┆ 1.447495      ┆ 1436.12        │
        │ Compass G-8 (Beidou IGSO-3)         ┆ 1.410042      ┆ 1435.93        │
        │ ...                                 ┆ ...           ┆ ...            │
        │ Navstar GPS II-24 (Navstar SVN 3... ┆ 0.670827      ┆ 716.69         │
        │ Navstar GPS IIR-10 (Navstar SVN ... ┆ 0.670764      ┆ 716.47         │
        │ Navstar GPS IIR-M-6 (Navstar SVN... ┆ 0.670744      ┆ 716.4          │
        │ Wind (International Solar-Terres... ┆ 0.546906      ┆ 19700.45       │
        └─────────────────────────────────────┴───────────────┴────────────────┘

        It looks like Intelsat 903 is the most inconsistent by a good amount.
        Let's take a look at it's data and see why its orbital period (very
        standard for a geosynchronos satellites) isn't consistent with the model.

        >>> cols = ['Period_minutes', 'Class_of_Orbit',
        ...         'Perigee_km', 'Apogee_km', 'Eccentricity']
        >>> engine.df.filter(pl.col('ID') == 'Intelsat 903')[cols].melt()
        shape: (5, 2)
        ┌────────────────┬────────────────────┐
        │ variable       ┆ value              │
        │ ---            ┆ ---                │
        │ str            ┆ str                │
        ╞════════════════╪════════════════════╡
        │ Period_minutes ┆ 1436.16            │
        │ Class_of_Orbit ┆ GEO                │
        │ Perigee_km     ┆ 35773.0            │
        │ Apogee_km      ┆ 358802.0           │
        │ Eccentricity   ┆ 0.7930699999999999 │
        └────────────────┴────────────────────┘
        """
        if given is None and (len(values.shape) == 1 or values.shape[1] == 1):
            raise ValueError(
                "If no `given` is provided more than one variable must be \
                provided in `values`"
            )

        logps = self.logp(values, given=given)
        if logps is None:
            return None

        if given is None:
            marg = sum([self.logp(values[col]) for col in values.columns])
        else:
            marg = self.logp(values)

        out = logps / marg

        if isinstance(out, pl.Series):
            out.rename("inconsistency")

        return out

    def surprisal(self, col: int | str, rows=None, values=None, state_ixs=None):
        """Compute the surprisal of a values in specific cells

        Surprisal is the negative log likeilihood of a specific value in a
        specific position (cell) in the table.

        Parameters
        ----------
        col: column index
            The column location of the target cells
        rows: arraylike[row index], optional
            Row indices of the cells. If ``None`` (default), all non-missing
            rows will be used.
        values: arraylike[value]
            Proposed values for each cell. Must have an entry for each entry in
            `rows`. If `None`, the existing values are used.

        Returns
        -------
        polars.DataFrame
            A polars.DataFrame containing an `index` column for the row names, a
            `<col>` column for the values, and a `surprisal` column containing
            the surprisal values.

        Examples
        --------

        Find satellites with the top five most surprising expected lifetimes

        >>> import polars as pl
        >>> from lace.examples import Satellites
        >>> engine = Satellites()
        >>> engine.surprisal('Expected_Lifetime') \\
        ...   .sort('surprisal', reverse=True) \\
        ...   .head(5)
        shape: (5, 3)
        ┌─────────────────────────────────────┬───────────────────┬───────────┐
        │ index                               ┆ Expected_Lifetime ┆ surprisal │
        │ ---                                 ┆ ---               ┆ ---       │
        │ str                                 ┆ f64               ┆ f64       │
        ╞═════════════════════════════════════╪═══════════════════╪═══════════╡
        │ International Space Station (ISS... ┆ 30.0              ┆ 6.312802  │
        │ Milstar DFS-5 (USA 164, Milstar ... ┆ 0.0               ┆ 5.470039  │
        │ Landsat 7                           ┆ 15.0              ┆ 5.385252  │
        │ Intelsat 701                        ┆ 0.5               ┆ 5.271304  │
        │ Optus B3                            ┆ 0.5               ┆ 5.271304  │
        └─────────────────────────────────────┴───────────────────┴───────────┘

        Compute the surprisal for specific cells

        >>> engine.surprisal(
        ...   'Expected_Lifetime',
        ...   rows=['Landsat 7', 'Intelsat 701']
        ... )
        shape: (2, 3)
        ┌──────────────┬───────────────────┬───────────┐
        │ index        ┆ Expected_Lifetime ┆ surprisal │
        │ ---          ┆ ---               ┆ ---       │
        │ str          ┆ f64               ┆ f64       │
        ╞══════════════╪═══════════════════╪═══════════╡
        │ Landsat 7    ┆ 15.0              ┆ 5.385252  │
        │ Intelsat 701 ┆ 0.5               ┆ 5.271304  │
        └──────────────┴───────────────────┴───────────┘

        Compute the surprisal of specific values in specific cells

        >>> engine.surprisal(
        ...   'Expected_Lifetime',
        ...   rows=['Landsat 7', 'Intelsat 701'],
        ...   values=[10.0, 10.0]
        ... )
        shape: (2, 3)
        ┌──────────────┬───────────────────┬───────────┐
        │ index        ┆ Expected_Lifetime ┆ surprisal │
        │ ---          ┆ ---               ┆ ---       │
        │ str          ┆ f64               ┆ f64       │
        ╞══════════════╪═══════════════════╪═══════════╡
        │ Landsat 7    ┆ 10.0              ┆ 3.198794  │
        │ Intelsat 701 ┆ 10.0              ┆ 2.530707  │
        └──────────────┴───────────────────┴───────────┘
        """
        return self.engine.surprisal(
            col, rows=rows, values=values, state_ixs=state_ixs
        )

    def simulate(
        self, cols, given=None, n: int = 1, include_given: bool = False
    ):
        """Simulate data from a conditional distribution

        Parameters
        ----------
        cols: list[column index]
            A list of target columns to simulate
        given: dict[column index, value], optional
            An optional dictionary of column -> value conditions
        n: int, optional
            The number of values to draw
        include_given: bool, optional
            If ``True``, the conditioning values in the given will be included
            in the output

        Returns
        -------
        polars.DataFrame
            The output data

        Examples
        --------

        Draw from a pair of columns

        >>> from lace.examples import Satellites
        >>> engine = Satellites()
        >>> engine.simulate(['Class_of_Orbit', 'Period_minutes'], n=5)
        shape: (5, 2)
        ┌────────────────┬────────────────┐
        │ Class_of_Orbit ┆ Period_minutes │
        │ ---            ┆ ---            │
        │ str            ┆ f64            │
        ╞════════════════╪════════════════╡
        │ LEO            ┆ 122.52184      │
        │ GEO            ┆ 1453.688835    │
        │ LEO            ┆ 127.016764     │
        │ MEO            ┆ 708.117944     │
        │ MEO            ┆ 4.09721        │
        └────────────────┴────────────────┘

        Simulate a pair of columns conditioned on another

        >>> engine.simulate(
        ...   ['Class_of_Orbit', 'Period_minutes'],
        ...   given={'Purpose': 'Communications'},
        ...   n=5
        ... )
        shape: (5, 2)
        ┌────────────────┬────────────────┐
        │ Class_of_Orbit ┆ Period_minutes │
        │ ---            ┆ ---            │
        │ str            ┆ f64            │
        ╞════════════════╪════════════════╡
        │ GEO            ┆ 1432.673621    │
        │ MEO            ┆ -86.757849     │
        │ LEO            ┆ 115.614145     │
        │ GEO            ┆ 1450.919225    │
        │ GEO            ┆ 1432.667778    │
        └────────────────┴────────────────┘

        Simulate missing values for columns that are missing not-at-random

        >>> engine.simulate(['longitude_radians_of_geo'], n=5)
        shape: (5, 1)
        ┌──────────────────────────┐
        │ longitude_radians_of_geo │
        │ ---                      │
        │ f64                      │
        ╞══════════════════════════╡
        │ null                     │
        │ -1.981454                │
        │ null                     │
        │ null                     │
        │ -0.333911                │
        └──────────────────────────┘
        >>> engine.simulate(
        ...   ['longitude_radians_of_geo'],
        ...   given={'Class_of_Orbit': 'GEO'},
        ...   n=5
        ... )
        shape: (5, 1)
        ┌──────────────────────────┐
        │ longitude_radians_of_geo │
        │ ---                      │
        │ f64                      │
        ╞══════════════════════════╡
        │ 2.413791                 │
        │ -0.666556                │
        │ 0.768952                 │
        │ -2.612664                │
        │ -0.895047                │
        └──────────────────────────┘

        If we simulate using ``given`` conditions, we can include the
        conditions in the output using ``include_given=True``.

        >>> engine.simulate(
        ...     ['Period_minutes'],
        ...     given={
        ...         'Purpose': 'Communications',
        ...         'Class_of_Orbit': 'GEO'
        ...     },
        ...     n=5,
        ...     include_given=True,
        ... )
        shape: (5, 3)
        ┌────────────────┬────────────────┬────────────────┐
        │ Period_minutes ┆ Purpose        ┆ Class_of_Orbit │
        │ ---            ┆ ---            ┆ ---            │
        │ f64            ┆ str            ┆ str            │
        ╞════════════════╪════════════════╪════════════════╡
        │ 1440.134814    ┆ Communications ┆ GEO            │
        │ 1436.590222    ┆ Communications ┆ GEO            │
        │ 1446.783909    ┆ Communications ┆ GEO            │
        │ 907.952479     ┆ Communications ┆ GEO            │
        │ 1431.973249    ┆ Communications ┆ GEO            │
        └────────────────┴────────────────┴────────────────┘
        """
        df = self.engine.simulate(cols, given=given, n=n)

        if include_given and given is not None:
            for k, v in given.items():
                col = pl.Series(k, [v] * n)
                df = df.with_columns(col)

        return df

    def draw(self, row: int | str, col: int | str, n: int = 1):
        """
        Draw data from the distribution of a specific cell in the table

        Draw differs from simulate in that it is derived from the distribution
        of at a specific cell in the table rather than a hypothetical

        Parameters
        ----------
        row: row index
           The row name or index of the cell
        col: column index
            The column name or index of the cell
        n: int, optional
            The number of samples to draw

        Returns
        -------
        polars.Series
            A polars Series with ``n`` draws from the cell at (row, col)

        Examples
        --------

        >>> from lace.examples import Satellites
        >>> engine = Satellites()
        >>> engine.draw('Landsat 7', 'Period_minutes', n=5)  # doctest: +NORMALIZE_WHITESPACE
        shape: (5,)
        Series: 'Period_minutes' [f64]
        [
            108.507463
            118.577182
            89.441123
            117.199444
            73.184567
        ]
        """
        srs = self.engine.draw(row, col, n)
        return utils.return_srs(srs)

    def predict(
        self,
        target: str | int,
        given: Optional[dict[str | int, object]] = None,
        with_uncertainty: bool = True,
    ):
        """Predict a single target from a conditional distribution

        Parameters
        ----------
        target: column index
            The column to predict
        given: dict[column index, value], optional
            Column -> Value dictionary describing observations. Note that
            columns can either be indices (int) or names (str)
        with_uncertainty: bool, optional
            if ``True`` (default), return the uncertainty

        Returns
        -------
        pred: value
            The predicted value
        unc: float, optional
            The uncertainty

        Examples
        --------

        Predict whether an animal swims and return uncertainty

        >>> from lace.examples import Animals
        >>> animals = Animals()
        >>> animals.predict('swims')
        (0, 0.008287057807910558)

        Predict whether an animal swims given that it has flippers

        >>> animals.predict('swims', given={'flippers': 1})
        (1, 0.05008037071634858)

        Let's confuse lace and see what happens to its uncertainty. Let's
        predict whether an non-water animal with flippers swims

        >>> animals.predict('swims', given={'flippers': 1, 'water': 0})
        (0, 0.32863593091906085)

        If you want to save time and you do not care about quantifying your
        epistemic uncertainty, you don't have to compute uncertainty.

        >>> animals.predict('swims', with_uncertainty=False)
        0
        """
        return self.engine.predict(target, given, with_uncertainty)

    def impute(
        self,
        col: str | int,
        rows: Optional[list[str | int]] = None,
        unc_type: Optional[str] = "js_divergence",
    ):
        """Impute (predict) the value of a cell(s) in the lace table

        Impute returns the most likely value at a specific location in the
        table. regardless of whether the cell at (``row``, ``col``) contains a
        present value, ``impute`` will choose the value that is most likely
        given the current distribution of the cell. If the current value is an
        outlier, or unlikely, ``impute`` will return a value that is more in
        line with its understanding of the data.

        If the cell lies in a missing-not-at-random column, a value will always
        be returned, even if the value is most likely to be missing. Imputation
        forces the value of a cell to be present.

        The following methods are used to compute uncertainty.

          * unc_type='js_divergence' computes the Jensen-Shannon divergence
            between the state imputation distributions.

            .. math::
              JS(X_1, X_2, ..., X_S)

          * unc_type='pairwise_kl' computes the mean of the Kullback-Leibler
            divergences between pairs of state imputation distributions.

            .. math::
              \\frac{1}{S^2 - S} \\sum_{i=1}^S \\sum{j \\in \\{1,..,S\\} \\setminus i} KL(X_i | X_j)

        Parameters
        ----------
        col: column index
            The column index
        rows: list[row index], optional
            Optional row indices to impute. If ``None`` (default), all the rows
            with missing values will be imputed
        unc_type: str, optional
            The type of uncertainty to compute. If ``None``, uncertainty will
            not be computed. Acceptable values are:
            * 'js_divergence' (default): The Jensen-Shannon divergence between the
              imputation distributions in each state.
            * 'pairwise_kl': The mean pairwise Kullback-Leibler divergence
              between pairs of state imputation distributions.

        Returns
        -------
        polars.DataFrame
            Indexed by ``rows``; contains a column for the imputed values and
            their uncertainties, if requested.

        Examples
        --------

        Impute, with uncertainty, all the missing values in a column

        >>> from lace.examples import Satellites
        >>> engine = Satellites()
        >>> engine.impute('Purpose')
        shape: (0, 2)
        ┌───────┬─────────┐
        │ index ┆ Purpose │
        │ ---   ┆ ---     │
        │ str   ┆ str     │
        ╞═══════╪═════════╡
        └───────┴─────────┘

        Let's choose a column that actually has missing values

        >>> engine.impute('Type_of_Orbit')  # doctest: +NORMALIZE_WHITESPACE
        shape: (645, 3)
        ┌─────────────────────────────────────┬─────────────────┬─────────────┐
        │ index                               ┆ Type_of_Orbit   ┆ uncertainty │
        │ ---                                 ┆ ---             ┆ ---         │
        │ str                                 ┆ str             ┆ f64         │
        ╞═════════════════════════════════════╪═════════════════╪═════════════╡
        │ AAUSat-3                            ┆ Sun-Synchronous ┆ 0.238266    │
        │ ABS-1 (LMI-1, Lockheed Martin-In... ┆ Sun-Synchronous ┆ 0.726554    │
        │ ABS-1A (Koreasat 2, Mugunghwa 2,... ┆ Sun-Synchronous ┆ 0.750425    │
        │ ABS-2i (MBSat, Mobile Broadcasti... ┆ Sun-Synchronous ┆ 0.727579    │
        │ ...                                 ┆ ...             ┆ ...         │
        │ Zhongxing 20A                       ┆ Sun-Synchronous ┆ 0.74625     │
        │ Zhongxing 22A (Chinastar 22A)       ┆ Sun-Synchronous ┆ 0.822414    │
        │ Zhongxing 2A (Chinasat 2A)          ┆ Sun-Synchronous ┆ 0.727579    │
        │ Zhongxing 9 (Chinasat 9, Chinast... ┆ Sun-Synchronous ┆ 0.727579    │
        └─────────────────────────────────────┴─────────────────┴─────────────┘

        Impute a defined set of rows

        >>> engine.impute('Purpose', rows=['AAUSat-3', 'Zhongxing 20A'])
        shape: (2, 3)
        ┌───────────────┬────────────────────────┬─────────────┐
        │ index         ┆ Purpose                ┆ uncertainty │
        │ ---           ┆ ---                    ┆ ---         │
        │ str           ┆ str                    ┆ f64         │
        ╞═══════════════╪════════════════════════╪═════════════╡
        │ AAUSat-3      ┆ Technology Development ┆ 0.209361    │
        │ Zhongxing 20A ┆ Communications         ┆ 0.04965     │
        └───────────────┴────────────────────────┴─────────────┘

        Uncertainty is optional

        >>> engine.impute('Type_of_Orbit', unc_type=None)
        shape: (645, 2)
        ┌─────────────────────────────────────┬─────────────────┐
        │ index                               ┆ Type_of_Orbit   │
        │ ---                                 ┆ ---             │
        │ str                                 ┆ str             │
        ╞═════════════════════════════════════╪═════════════════╡
        │ AAUSat-3                            ┆ Sun-Synchronous │
        │ ABS-1 (LMI-1, Lockheed Martin-In... ┆ Sun-Synchronous │
        │ ABS-1A (Koreasat 2, Mugunghwa 2,... ┆ Sun-Synchronous │
        │ ABS-2i (MBSat, Mobile Broadcasti... ┆ Sun-Synchronous │
        │ ...                                 ┆ ...             │
        │ Zhongxing 20A                       ┆ Sun-Synchronous │
        │ Zhongxing 22A (Chinastar 22A)       ┆ Sun-Synchronous │
        │ Zhongxing 2A (Chinasat 2A)          ┆ Sun-Synchronous │
        │ Zhongxing 9 (Chinasat 9, Chinast... ┆ Sun-Synchronous │
        └─────────────────────────────────────┴─────────────────┘
        """
        return self.engine.impute(col, rows, unc_type)

    def depprob(self, col_pairs: list):
        """Compute the dependence probability between pairs of columns

        The dependence probability between columns X and Y is the probability
        that a dependence path exists between two columns. If X is predictive of
        Y (or the reverse), dependence probability will be closer to 1.

        The dependence probability between two columns is defined as the
        proportion of lace states in which those two columns belong to the same
        view.

        Parameters
        ----------
        col_pairs: list((column index, column index))
            A list of pairs of columns for which to compute dependence
            probability

        Returns
        -------
        float, polars.Series
            Contains a entry for each pair in ``col_pairs``. If ``col_pairs``
            contains a single entry, a float will be returned.

        Notes
        -----
        Note that high dependence probability does not always indicate that two
        variables are mutually predictive. For example in the model

        X ~ Normal(0, 1)
        Y ~ Normal(0, 1)
        Z ~ X + Y

        X and Y are completely independent of each other, by X and Y are
        predictable through Z. If you know X and Z, you know Y. In this case X
        and Y will have a high dependence probability because of their shared
        relationship with Z.

        If you are only interested in the magnitude of predictive power between
        two variables, use mutual information via the ``mi`` function.

        See Also
        --------
        mi

        Examples
        --------

        A single pair as input gets you a float output

        >>> from lace.examples import Animals
        >>> engine = Animals()
        >>> engine.depprob([('swims', 'flippers')])
        1.0

        Multiple pairs as inputs gets you a polars ``Series``

        >>> engine.depprob([
        ...   ('swims', 'flippers'),
        ...   ('fast', 'tail'),
        ... ])  # doctest: +NORMALIZE_WHITESPACE
        shape: (2,)
        Series: 'depprob' [f64]
        [
            1.0
            0.625
        ]
        """
        srs = self.engine.depprob(col_pairs)
        return utils.return_srs(srs)

    def mi(
        self, col_pairs: list, n_mc_samples: int = 1000, mi_type: str = "iqr"
    ):
        """Compute the mutual information between pairs of columns

        The mutual information is the amount of information (in nats) between
        two variables.

        Parameters
        ----------
        col_pairs: list((column index, column index))
            A list of pairs of columns for which to compute mutual information
        n_mc_samples: int
            The number of samples to use when Monte Carlo integration is used to
            approximate mutual information. More samples gives you less error,
            but takes longer.
        mi_type: str
            The variant of mutual information to compute. Different variants
            normalize to within a range and give different behavior. See Notes
            for more information on the supported variants.

        Returns
        -------
        float, polars.Series
            Contains a entry for each pair in ``col_pairs``. If ``col_pairs``
            contains a single entry, a float will be returned.

        Notes
        -----
        Supported Variants:
            - 'unnormed': standard, un-normalized mutual information
            - 'normed': normalized by the minimum of the two variables'
              entropies, e.g. `min(H(X), H(Y))`, which scales mutual information
              to the interval [0, 1]
            - 'linfoot': A variation of mutual information derived by solving
              for the correlation coefficient between two components of a
              bivariate normal distribution with given mutual information
            - 'voi': Variation of Information. A version of mutual information
              that satisfies the triangle inequality.
            - 'jaccard': the Jaccard distance between two variables is 1-VOI
            - 'iqr': Information Quality Ratio. The amount of information of a
              variable based on another variable against total uncertainty.
            - 'pearson': mutual information normalized by the square root of the
              product of the component entropies, ``sqrt(H(X)*H(Y))``. Akin to
              the Pearson correlation coefficient.

        Note that mutual information may misbehave for continuous variables
        because entropy can be negative for continuous variables (see
        differential entropy). If this is likely to be an issue, use the
        'linfoot' ``mi_type`` or use ``depprob``.

        Examples
        --------

        A single pair as input gets you a float output

        >>> from lace.examples import Animals
        >>> engine = Animals()
        >>> engine.mi([('swims', 'flippers')])
        0.27197816458827445

        You can select different normalizations of mutual information

        >>> engine.mi([('swims', 'flippers')], mi_type='unnormed')
        0.19361180218629537

        Multiple pairs as inputs gets you a polars ``Series``

        >>> engine.mi([
        ...   ('swims', 'flippers'),
        ...   ('fast', 'tail'),
        ... ])  # doctest: +NORMALIZE_WHITESPACE
        shape: (2,)
        Series: 'mi' [f64]
        [
            0.271978
            0.005378
        ]
        """
        srs = self.engine.mi(
            col_pairs, n_mc_samples=n_mc_samples, mi_type=mi_type
        )
        return utils.return_srs(srs)

    def rowsim(
        self,
        row_pairs: list,
        wrt: Optional[list] = None,
        col_weighted: bool = False,
    ):
        """Compute the row similarity between pairs of rows

        Row similarity (or relevance) takes on continuous values in [0, 1] and
        is a measure of how similar two rows are with respect to how their
        values are modeled. This is distinct from distance-based measures in
        that it looks entirely in model space. This has a number of advantages
        such as scaling independent of the data (or even the data types) and
        complete disregard for missing values (all cells, missing or occupied,
        are assigned to a category).

        The row similarity between two rows, A and B, is defined as the mean
        proportion of categories in which the two rows are in the same category.

        Parameters
        ----------
        row_pairs: list[(row index, row index)]
            A list of row pairs for which to compute row similarity
        wrt: list[column index], optional
            An optional list of column indices to provide context. If columns
            are provided via ``wrt``, only views containing these columns will
            be considered in the row similarity computation. If ``None``
            (default), all views are considered.
        col_weighted: bool
            If ``True``, row similarity will compute the proportion of relevant
            columns, instead of views, in which the two rows are in the same
            category.

        Returns
        -------
        float, polars.Series
            Contains a entry for each pair in ``row_pairs``. If ``row_pairs``
            contains a single entry, a float will be returned.

        Examples
        --------

        How similar are a beaver and a polar bear?

        >>> from lace.examples import Animals
        >>> animals = Animals()
        >>> animals.rowsim([('beaver', 'polar+bear')])
        0.6059523809523808

        What about if we weight similarity by columns and not the standard
        views?

        >>> animals.rowsim([('beaver', 'polar+bear')], col_weighted=True)
        0.5698529411764706

        Not much change. How similar are they with respect to how we model their
        swimming?

        >>> animals.rowsim([('beaver', 'polar+bear')], wrt=['swims'])
        0.875

        Very similar. But will all animals that swim be highly similar with
        respect to their swimming?

        >>> animals.rowsim([('otter', 'polar+bear')], wrt=['swims'])
        0.375

        Lace predicts an otter's swimming for different reasons than a polar
        bear's.

        What is a Chihuahua more similar to, a wolf or a rat?

        >>> from lace.examples import Animals
        >>> engine = Animals()
        >>> engine.rowsim([
        ...   ('chihuahua', 'wolf'),
        ...   ('chihuahua', 'rat'),
        ... ]) # doctest: +NORMALIZE_WHITESPACE
        shape: (2,)
        Series: 'rowsim' [f64]
        [
            0.629315
            0.772545
        ]
        """
        srs = self.engine.rowsim(row_pairs, wrt=wrt, col_weighted=col_weighted)
        return utils.return_srs(srs)

    def novelty(self, row, wrt=None):
        """
        Compute the novelty of a row
        """
        return self.engine.novelty(row, wrt)

    def pairwise_fn(self, fn_name, indices: Optional[list] = None, **kwargs):
        """Compute a function for a set of pairs of rows or columns

        Parameters
        ----------
        fn_name: str
            The name of the function: 'rowsim', 'mi', or 'depprob'
        indices: list[index], optional
            An optional list of indices from which to generate pairs. The output
            will be the function computed over the Cartesian product of
            ``indices``. If ``None`` (default), all indices will be considered.

        All other keyword arguments will be passed to the target function

        Examples
        --------

        Column weighted row similarity with indices defined

        >>> from lace.examples import Animals
        >>> engine = Animals()
        >>> engine.pairwise_fn(
        ...   'rowsim',
        ...   indices=['wolf', 'rat', 'otter'],
        ... )
        shape: (9, 3)
        ┌───────┬───────┬──────────┐
        │ A     ┆ B     ┆ rowsim   │
        │ ---   ┆ ---   ┆ ---      │
        │ str   ┆ str   ┆ f64      │
        ╞═══════╪═══════╪══════════╡
        │ wolf  ┆ wolf  ┆ 1.0      │
        │ wolf  ┆ rat   ┆ 0.71689  │
        │ wolf  ┆ otter ┆ 0.492262 │
        │ rat   ┆ wolf  ┆ 0.71689  │
        │ ...   ┆ ...   ┆ ...      │
        │ rat   ┆ otter ┆ 0.613095 │
        │ otter ┆ wolf  ┆ 0.492262 │
        │ otter ┆ rat   ┆ 0.613095 │
        │ otter ┆ otter ┆ 1.0      │
        └───────┴───────┴──────────┘

        Extra keyword arguments are passed to the parent function.

        >>> engine.pairwise_fn(
        ...   'rowsim',
        ...   indices=['wolf', 'rat', 'otter'],
        ...   col_weighted=True,
        ... )
        shape: (9, 3)
        ┌───────┬───────┬──────────┐
        │ A     ┆ B     ┆ rowsim   │
        │ ---   ┆ ---   ┆ ---      │
        │ str   ┆ str   ┆ f64      │
        ╞═══════╪═══════╪══════════╡
        │ wolf  ┆ wolf  ┆ 1.0      │
        │ wolf  ┆ rat   ┆ 0.642647 │
        │ wolf  ┆ otter ┆ 0.302206 │
        │ rat   ┆ wolf  ┆ 0.642647 │
        │ ...   ┆ ...   ┆ ...      │
        │ rat   ┆ otter ┆ 0.491176 │
        │ otter ┆ wolf  ┆ 0.302206 │
        │ otter ┆ rat   ┆ 0.491176 │
        │ otter ┆ otter ┆ 1.0      │
        └───────┴───────┴──────────┘

        If you do not provide indices, the function is computed for the product
        of all indices.

        >>> engine.pairwise_fn('rowsim')
        shape: (2500, 3)
        ┌──────────┬──────────────┬──────────┐
        │ A        ┆ B            ┆ rowsim   │
        │ ---      ┆ ---          ┆ ---      │
        │ str      ┆ str          ┆ f64      │
        ╞══════════╪══════════════╪══════════╡
        │ antelope ┆ antelope     ┆ 1.0      │
        │ antelope ┆ grizzly+bear ┆ 0.464137 │
        │ antelope ┆ killer+whale ┆ 0.479613 │
        │ antelope ┆ beaver       ┆ 0.438467 │
        │ ...      ┆ ...          ┆ ...      │
        │ dolphin  ┆ walrus       ┆ 0.724702 │
        │ dolphin  ┆ raccoon      ┆ 0.340923 │
        │ dolphin  ┆ cow          ┆ 0.482887 │
        │ dolphin  ┆ dolphin      ┆ 1.0      │
        └──────────┴──────────────┴──────────┘
        """
        if indices is not None:
            pairs = list(it.product(indices, indices))
        else:
            pairs, _ = utils.get_all_pairs(fn_name, self.engine)

        return self.engine.pairwise_fn(fn_name, pairs, kwargs)

    def clustermap(
        self,
        fn_name: str,
        *,
        indices=None,
        linkage_method="ward",
        no_plot=False,
        fn_kwargs={},
        **kwargs,
    ) -> ClusterMap:
        """Generate a clustermap of a pairwise function

        Parameters
        ----------
        fn_name: str
            The name of the function: 'rowsim', 'mi', or 'depprob'
        indices: list[index], optional
            An optional list of indices from which to generate pairs. The output
            will be the function computed over the Cartesian product of
            ``indices``. If ``None`` (default), all indices will be considered.
        linkage_method: str, optional
            The linkage method for computing the hierarchical clustering over
            the pairwise function values. This values is passed to
            [``scipy.cluster.hierarchy.linkage``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage)
        no_plot: bool, optional
            If true, returns the linkage, and clustered pairwise function, but
            does not build a plot
        fn_kwargs: dict, optional
            Keyword arguments passed to the target function

        All other arguments passed to plotly ``imshow``

        Returns
        -------
        ClusterMap
            The result of the computation. Contains the following fields:
            - df: the clusterd polars.DataFrame computed by ``pairwise_fn``
            - linkage: the scipy-generated linkage
            - figure (optional): the plotly figure

        Examples
        --------

        Compute a dependence probability clustermap

        >>> from lace.examples import Animals
        >>> animals = Animals()
        >>> animals.clustermap(
        ...   'depprob',
        ...   zmin=0,
        ...   zmax=1,
        ...   color_continuous_scale='greys'
        ... ).figure.show()

        Use the ``fn_kwargs`` keyword argument to pass keyword arguments to the
        target function.

        >>> animals.clustermap(
        ...   'rowsim',
        ...   zmin=0,
        ...   zmax=1,
        ...   color_continuous_scale='greys',
        ...   fn_kwargs={'wrt': ['swims']},
        ... ).figure.show()
        """
        fn = self.pairwise_fn(fn_name, indices, **fn_kwargs)

        df = fn.pivot(values=fn_name, index="A", columns="B")
        df, linkage = utils.hcluster(df, method=linkage_method)

        if not no_plot:
            fig = px.imshow(
                df[:, 1:],
                labels=dict(x="A", y="B", color=fn_name),
                y=df["A"],
                **kwargs,
            )
            return ClusterMap(df, linkage, fig)
        else:
            return ClusterMap(df, linkage)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
