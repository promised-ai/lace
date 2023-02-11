import itertools as it
from typing import Optional
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
        if 'metadata' in kwargs:
            self.engine = lace_core.CoreEngine.load(kwargs['metadata'])
        else:
            self.engine = lace_core.CoreEngine(**args, **kwargs)

    @property
    def shape(self):
        return self.engine.shape

    @property
    def n_rows(self):
        return self.engine.n_rows

    @property
    def n_cols(self):
        return self.engine.n_cols

    @property
    def n_states(self):
        return self.engine.n_states

    @property
    def columns(self):
        return self.engine.columns

    @property
    def index(self):
        return self.engine.index

    @property
    def ftypes(self):
        return self.engine.ftypes

    def ftype(self, col):
        return self.engine.ftype(col)

    def __getitem__(self, ix):
        return self.engine[ix]

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
            return self.engine.logp_scaled(values, given, col_max_logps)
        else:
            return self.engine.logp(values, given)

    def inconsistency(self, values, given=None):
        '''Compute inconsistency
        '''
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


    def simulate(
        self,
        cols,
        given=None,
        n:int=1,
        include_given: bool=False
    ):
        '''Simulate data from a conditional distribution

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
        '''
        df = self.engine.simulate(cols, given=given, n=n)

        if include_given and given is not None:
            for k, v in given.items():
                df[k] = v

        return df

    def draw(self, row, col, n:int=1):
         '''Draw data from the distribution of a specific cell in the table

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
             A polars Series with `n` draws from the cell at (row, col)

        Examples
        --------
        '''
         raise NotImplementedError

    def predict(
        self,
        target,
        given: Optional[dict]=None,
        with_uncertainty=True
    ):
        ''' Predict a single target from a conditional distribution
    
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
        '''

    def impute(
        self,
        col,
        rows: Optional[list]=None,
        unc_type:Optional[str]=None
    ):
        '''Impute (predict) the value of a cell in the lace table
        '''
        raise NotImplementedError

    def depprob(self, col_pairs: list):
        '''Compute the dependence probability between pairs of columns

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

        >>> from lace import Engine
        >>> engine = Engine(example='animals')
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
        '''
        return self.engine.depprob(col_pairs)

    def mi(self, col_pairs: list, n_mc_samples: int=1000, mi_type: str='iqr'):
        '''Compute the mutual information between pairs of columns

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

        >>> from lace import Engine
        >>> engine = Engine(example='animals')
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

        '''
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
        '''Compute the row similarity between pairs of rows

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

        >>> from lace import Engine
        >>> engine = Engine(example='animals')
        >>> engine.rowsim([
        ...   ('chihuahua', 'wolf'),
        ...   ('chihuahua', 'rat'),
        ... ])
        '''
        return self.engine.rowsim(row_pairs, wrt=wrt, col_weighted=col_weighted)

    def pairwise_fn(self, fn_name, indices:Optional[list]=None, **kwargs):
        '''Compute a function for a set of pairs of rows or columns

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

        >>> from lace import Engine
        >>> engine = Engine(example='animals')
        >>> engine.pairwise_fn(
        ...   'rowsim',
        ...   indices=['wolf', 'rat', 'otter'],
        ...   col_weighted=True,
        ... )
        '''
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
        '''Generate a clustermap of a pairwise function

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
        '''
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
