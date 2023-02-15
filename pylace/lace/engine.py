import itertools as it
from typing import Optional
from pathlib import Path
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
        """
        if 'metadata' in kwargs:
            if len(kwargs) > 1:
                raise ValueError("No other arguments may be privded if \
                                 `metadata` is provided")
            self.engine = lace_core.CoreEngine.load(kwargs['metadata'])
        else:
            self.engine = lace_core.CoreEngine(*args, **kwargs)

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
        32
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
        >>> engine.columns
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
        return self.engine.index

    @property
    def ftypes(self):
        """
        A dictionary mapping column names to feature types

        Examples
        --------
        >>> from lace.examples import Satellites
        >>> engine = Satellites()
        >>> engine.ftypes
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

    def ftype(self, col):
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

    def __getitem__(self, ix):
        return self.engine[ix]

    def save(self, path: Path):
        """Save the Engine metadata to ``path``
        """
        raise NotImplementedError

    def append_rows(self, rows):
        self.engine.append_rows(rows)

    def update(
        self,
        n_iters,
        *,
        timeout=None,
        checkpoint=None,
        transitions=None,
        save_path=None,
    ):
        return self.engine.update(
            n_iters,
            timeout=timeout,
            checkpoint=checkpoint,
            transitions=transitions,
            save_path=save_path
        )

    def entropy(self, cols, n_mc_samples: int=1000):
        return self.engine.entropy(cols, n_mc_samples)

    def logp(self, values, given=None, *, scaled=False, col_max_logps=None):
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
        """
        if scaled:
            # TODO: add a class method to compute the cache
            return self.engine.logp_scaled(values, given, col_max_logps)
        else:
            return self.engine.logp(values, given)

    def inconsistency(self, values, given=None):
        """Compute inconsistency
        """
        logps = self.logp(values, given=given)
        if given is None:
            pass
            marg = sum([self.logp(values[col]) for col in values.columns])
        else:
            marg = self.logp(values)

        out = logps / marg

        if isinstance(out, pl.Series):
            out.rename('inconsistency')

        return out

    def surprisal(self, col, rows=None, values=None, state_ixs=None):
        """Compute the surprisal of a values in specific cells

        Surprisal is the negative log likeilihood of a specific value in a
        specific position (cell) in the table.

        Parameters
        ----------
        col: column index
            The column location of the target cells
        rows: list[row index], optional
            Row indices of the cells. If ``None`` (default), all non-missing
            rows will be used.
        values: list[value}
        """
        return self.engine.surprisal(
            col, rows=rows, values=values, state_ixs=state_ixs)


    def simulate(
        self,
        cols,
        given=None,
        n:int=1,
        include_given: bool=False
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
        """
        df = self.engine.simulate(cols, given=given, n=n)

        if include_given and given is not None:
            for k, v in given.items():
                df[k] = v

        return df

    def draw(self, row, col, n: int=1):
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
        """
        return self.engine.draw(row, col, n)

    def predict(
        self,
        target,
        given: Optional[dict]=None,
        with_uncertainty=True
    ):
        """ Predict a single target from a conditional distribution

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
        """
        return self.engine.predict(target, given, with_uncertainty)

    def impute(
        self,
        col,
        rows: Optional[list]=None,
        unc_type: Optional[str]='js_divergence',
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


        Impute a defined set of rows
        >>> engine.impute('Purpose', rows=['FIXME'])

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
        ...   ('fast', 'slow'),
        ... ])


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
        """
        return self.engine.depprob(col_pairs)

    def mi(self, col_pairs: list, n_mc_samples: int=1000, mi_type: str='iqr'):
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

        Examples
        --------

        A single pair as input gets you a float output

        >>> from lace.examples import Animals
        >>> engine = Animals()
        >>> engine.mi([('swims', 'flippers')])

        Multiple pairs as inputs gets you a polars ``Series``

        >>> engine.mi([
        ...   ('swims', 'flippers'),
        ...   ('fast', 'swims'),
        ... ])

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

        """
        return self.engine.mi(
            col_pairs,
            n_mc_samples=n_mc_samples,
            mi_type=mi_type
        )

    def rowsim(
        self,
        row_pairs: list,
        wrt: Optional[list]=None,
        col_weighted: bool=False
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

        What is a Chihuahua more similar to, a wolf or a rat?

        >>> from lace.examples import Animals
        >>> engine = Animals()
        >>> engine.rowsim([
        ...   ('chihuahua', 'wolf'),
        ...   ('chihuahua', 'rat'),
        ... ])
        """
        return self.engine.rowsim(row_pairs, wrt=wrt, col_weighted=col_weighted)

    def novelty(self, row, wrt=None):
        return self.engine.novelty(row, wrt)

    def pairwise_fn(self, fn_name, indices:Optional[list]=None, **kwargs):
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
        ...   col_weighted=True,
        ... )
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
        linkage_method='ward',
        no_plot=False,
        fn_kwargs={},
        **kwargs
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
        """
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
