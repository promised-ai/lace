"""The main interface to Lace models."""
import itertools as it
from os import PathLike
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import pandas as pd
import plotly.express as px
import polars as pl

from lace import core, utils
from lace.codebook import Codebook
from lace.core import CodebookBuilder

if TYPE_CHECKING:
    import numpy as np


class ClusterMap:
    """
    Contains information about a pairwise function computed over a number of values.

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

    def __init__(self, df: pl.DataFrame, linkage: "np.ndarray", figure=None):
        self.df = df
        self.figure = figure
        self.linkage = linkage


class Engine:
    """The cross-categorization model with states and data."""

    engine: core.CoreEngine

    def __init__(self, core_engine: core.CoreEngine) -> None:
        """
        Create a new ``Engine`` with its internal representation.

        In general, you will use ``Engine.from_df`` or ``Engine.load``
        instead.
        """
        self.engine = core_engine

    @classmethod
    def from_df(
        cls,
        df: Union[pd.DataFrame, pl.DataFrame],
        codebook: Optional[
            Union[CodebookBuilder, PathLike, str, Codebook]
        ] = None,
        n_states: int = 8,
        id_offset: int = 0,
        rng_seed: Optional[int] = None,
    ) -> "Engine":
        """
        Create a new ``Engine`` from a DataFrame.

        Parameters
        ----------
        dataframe: pd.DataFrame or pl.DataFrame
            DataFrame with relevant data.
        codebook: CodebookBuilder or PathLike or str, optional
            Codebook builder which can load codebook from file or generate one
            from data. See ``CodebookBuilder``.
        n_states: int, optional
            The number of states (independent Markov chains).
        id_offset: int, optional
            An offset for renaming states in the metadata. Used when training a
            single engine on multiple machines. If one wished to split an
            8-state ``Engine`` run on to two machine, one may run a 4-state
            ``Engine`` on the first machine, then a 4-state ``Engine`` on the
            second machine with ``id_offset=4``. The states within two metadata
            files may be merged by copying without name collisions.
        rng_seed: int, optional
            Random number generator seed.


        Examples
        --------
        Create a new ``Engine`` from a DataFrame

        >>> from lace import Engine
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...    "ID": [1, 2, 3, 4],
        ...    "list_b": [2.0, 4.0, 6.0, 8.0],
        ... })
        >>> engine = Engine.from_df(df)

        Create a new ``Engine`` with specific codebook inference rules
        >>> from lace import Engine, CodebookBuilder
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...    "ID": [1, 2, 3, 4],
        ...    "list_b": [2.0, 4.0, 6.0, 8.0],
        ... })
        >>> engine = Engine.from_df(df, CodebookBuilder.infer(
        ...     cat_cutoff=2,
        ... ))

        """
        if isinstance(df, pd.DataFrame):
            df.index.rename("ID", inplace=True)
            df = pl.from_pandas(df, include_index=True)

        if codebook is not None:
            if isinstance(codebook, (str, PathLike)):
                codebook = CodebookBuilder.load(codebook)
            elif isinstance(codebook, Codebook):
                codebook = CodebookBuilder.codebook(codebook.codebook)

        return cls(
            core.CoreEngine(
                df,
                codebook,
                n_states,
                id_offset,
                rng_seed,
            )
        )

    @classmethod
    def load(cls, path: Union[str, bytes, PathLike]) -> "Engine":
        """
        Load an Engine from a path.

        Parameters
        ----------
        path: PathLike
            Path to the serialized ``Engine``.

        Examples
        --------
        Load an Engine from metadata

        >>> from lace import Engine  # doctest: +SKIP
        >>> engine = Engine.load("metadata.lace")  # doctest: +SKIP
        """
        return cls(core.CoreEngine.load(path))

    def save(self, path: Union[str, bytes, PathLike]):
        """
        Save the Engine metadata to ``path``.

        Examples
        --------
        Save a copy of an engine

        >>> from lace import Engine  # doctest: +SKIP
        >>> engine = Engine(metadata="metadata.lace")  # doctest: +SKIP
        >>> engine.save("metadata-copy.lace")  # doctest: +SKIP
        """
        self.engine.save(path)

    def seed(self, rng_seed: int):
        """
        Set the state of the random number generator (RNG).

        Parameters
        ----------
        rng_seed: int
            The desired state of the RNG

        Examples
        --------
        Re-simulate the same data.

        >>> from lace.examples import Animals
        >>> animals = Animals()
        >>> animals.seed(1337)
        >>> xs = animals.simulate(["swims", "slow", "flippers", "black"], n=100)
        >>> ys = animals.simulate(["swims", "slow", "flippers", "black"], n=100)
        >>> # count the number of cells that are different in each column
        >>> (xs != ys).sum()
        shape: (1, 4)
        ┌───────┬──────┬──────────┬───────┐
        │ swims ┆ slow ┆ flippers ┆ black │
        │ ---   ┆ ---  ┆ ---      ┆ ---   │
        │ u32   ┆ u32  ┆ u32      ┆ u32   │
        ╞═══════╪══════╪══════════╪═══════╡
        │ 34    ┆ 49   ┆ 20       ┆ 49    │
        └───────┴──────┴──────────┴───────┘

        If we set the seed, we get the same data.

        >>> animals.seed(1337)
        >>> zs = animals.simulate(["swims", "slow", "flippers", "black"], n=100)
        >>> # count the number of cells that are different in each column
        >>> (xs != zs).sum()
        shape: (1, 4)
        ┌───────┬──────┬──────────┬───────┐
        │ swims ┆ slow ┆ flippers ┆ black │
        │ ---   ┆ ---  ┆ ---      ┆ ---   │
        │ u32   ┆ u32  ┆ u32      ┆ u32   │
        ╞═══════╪══════╪══════════╪═══════╡
        │ 0     ┆ 0    ┆ 0        ┆ 0     │
        └───────┴──────┴──────────┴───────┘
        """
        self.engine.seed(rng_seed)

    @property
    def shape(self):
        """
        A tuple containing the number of rows and the number of columns in the table.

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
        The number of rows in the table.

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
        The number of columns in the table.

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
        The number of states (independent Markov chains).

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
        A list of the column names appearing in their order in the table.

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
        """The string row names of the engine."""
        return self.engine.index

    @property
    def ftypes(self):
        """
        A dictionary mapping column names to feature types.

        Examples
        --------
        >>> from lace.examples import Satellites  # doctest: +SKIP
        >>> engine = Satellites()  # doctest: +SKIP
        >>> engine.ftypes  # doctest: +SKIP
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

    @property
    def codebook(self) -> Codebook:
        """
        Return the codebook.

        Note that mutating the codebook will not affect the engine.
        """
        return Codebook(self.engine.codebook)

    def ftype(self, col: Union[str, int]):
        """
        Get the feature type of a column.

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
        >>> engine.ftype("Class_of_Orbit")
        'Categorical'
        >>> engine.ftype("Period_minutes")
        'Continuous'
        """
        return self.engine.ftype(col)

    def flatten_columns(self):
        """
        Flatten the column assignment.

        The resulting states will all have one view.

        Examples
        --------
        >>> from lace.examples import Satellites
        >>> engine = Satellites()
        >>> engine.column_assignment(0)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        >>> engine.column_assignment(1)
        [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 0, 0, 2, 2, 2, 2, 0]
        >>> engine.flatten_columns()
        >>> engine.column_assignment(0)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        >>> engine.column_assignment(1)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        >>> all(sum(engine.column_assignment(i)) == 0 for i in range(engine.n_states))
        True

        """
        self.engine.flatten_columns()

    def column_assignment(self, state_ix: int) -> List[int]:
        """
        Return the assignment of columns to views.

        Parameters
        ----------
        state_ix: int
            The state index for which to pull the column assignment

        Returns
        -------
        asgn: List[int]
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
        Return the assignment of rows to categories for each view.

        Parameters
        ----------
        state_ix: int
            The state index for which to pull the column assignment

        Returns
        -------
        asgn: List[List[int]]
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

    def __getitem__(self, ix):
        df = self.engine[ix]
        if df.shape[0] == 1 and df.shape[1] == 2:
            return df[0, 1]
        else:
            return df

    def diagnostics(self, name: str = "score"):
        """
        Get convergence diagnostics.

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
        shape: (5_000, 4)
        ┌──────────────┬──────────────┬──────────────┬──────────────┐
        │ score_0      ┆ score_1      ┆ score_2      ┆ score_3      │
        │ ---          ┆ ---          ┆ ---          ┆ ---          │
        │ f64          ┆ f64          ┆ f64          ┆ f64          │
        ╞══════════════╪══════════════╪══════════════╪══════════════╡
        │ -2882.424453 ┆ -2809.0876   ┆ -2638.714156 ┆ -2604.137622 │
        │ -2695.299327 ┆ -2666.497867 ┆ -2608.185358 ┆ -2576.545684 │
        │ -2642.539971 ┆ -2532.638368 ┆ -2576.463401 ┆ -2568.516617 │
        │ -2488.369418 ┆ -2513.134161 ┆ -2549.299382 ┆ -2554.131179 │
        │ …            ┆ …            ┆ …            ┆ …            │
        │ -1972.005746 ┆ -2122.788121 ┆ -1965.921104 ┆ -1969.328651 │
        │ -1966.516529 ┆ -2117.398333 ┆ -1993.351756 ┆ -1986.589833 │
        │ -1969.400394 ┆ -2147.941128 ┆ -1968.697139 ┆ -1988.805311 │
        │ -1920.217666 ┆ -2081.368421 ┆ -1909.655836 ┆ -1920.432849 │
        └──────────────┴──────────────┴──────────────┴──────────────┘
        """
        diag = self.engine.diagnostics()

        srss = []
        for ix in range(self.n_states):
            srs = utils._diagnostic(name, diag[ix])
            srss.append(srs.rename(f"{name}_{ix}"))

        # we have to align the engine diagnostics so they fit in the dataframe.
        # To do this, we pad the ends with None.
        max_len = max(len(srs) for srs in srss)
        diffs = [max_len - len(srs) for srs in srss]
        srss = [srs.extend_constant(None, n=n) for srs, n in zip(srss, diffs)]

        return pl.DataFrame(srss)

    def edit_cell(self, row: Union[str, int], col: Union[str, int], value):
        r"""
        Edit the value of a cell in the table.

        Parameters
        ----------
        row: row index
            The row index of the cell to edit
        col: column index
            The column index of the cell to edit
        value: value
            The new value at the cell

        Examples
        --------
        Change a surprising value

        >>> from lace.examples import Animals
        >>> animals = Animals()
        >>> # top five most surprisingly fierce animals
        >>> animals.surprisal('fierce') \
        ...     .sort('surprisal', descending=True) \
        ...     .head(5)
        shape: (5, 3)
        ┌────────────┬────────┬───────────┐
        │ index      ┆ fierce ┆ surprisal │
        │ ---        ┆ ---    ┆ ---       │
        │ str        ┆ u8     ┆ f64       │
        ╞════════════╪════════╪═══════════╡
        │ pig        ┆ 1      ┆ 1.565845  │
        │ rhinoceros ┆ 1      ┆ 1.094639  │
        │ buffalo    ┆ 1      ┆ 1.094639  │
        │ chihuahua  ┆ 1      ┆ 0.802085  │
        │ chimpanzee ┆ 1      ┆ 0.723817  │
        └────────────┴────────┴───────────┘
        >>>  # change  pig to not fierce
        >>> animals.edit_cell('pig', 'fierce', 0)
        >>> animals.surprisal('fierce') \
        ...     .sort('surprisal', descending=True) \
        ...     .head(5)
        shape: (5, 3)
        ┌────────────┬────────┬───────────┐
        │ index      ┆ fierce ┆ surprisal │
        │ ---        ┆ ---    ┆ ---       │
        │ str        ┆ u8     ┆ f64       │
        ╞════════════╪════════╪═══════════╡
        │ rhinoceros ┆ 1      ┆ 1.094639  │
        │ buffalo    ┆ 1      ┆ 1.094639  │
        │ chihuahua  ┆ 1      ┆ 0.802085  │
        │ chimpanzee ┆ 1      ┆ 0.723817  │
        │ dalmatian  ┆ 0      ┆ 0.594919  │
        └────────────┴────────┴───────────┘

        Set a value to missing

        >>> animals.edit_cell('pig', 'fierce', None)
        >>> # by default impute fills computes only missing values
        >>> animals.impute('fierce')
        shape: (1, 3)
        ┌───────┬────────┬─────────────┐
        │ index ┆ fierce ┆ uncertainty │
        │ ---   ┆ ---    ┆ ---         │
        │ str   ┆ u8     ┆ f64         │
        ╞═══════╪════════╪═════════════╡
        │ pig   ┆ 0      ┆ 0.02385     │
        └───────┴────────┴─────────────┘
        """
        self.engine.edit_cell(row, col, value)

    def append_rows(
        self,
        rows: Union[
            pd.Series, pd.DataFrame, pl.DataFrame, Dict[str, Dict[str, object]]
        ],
    ):
        """
        Append new rows to the table.

        Parameters
        ----------
        rows: polars.DataFrame, pandas.DataFrame, pandas.Series, Dict[str, dict]
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
        >>> crab_and_sponge = pl.DataFrame(
        ...     {
        ...         "index": ["crab", "sponge"],
        ...         "water": [1, 1],
        ...         "flippers": [0, 0],
        ...     }
        ... )
        >>> engine.append_rows(crab_and_sponge)
        >>> engine.index[-1]
        'sponge'
        >>> engine[-1, "water"]
        1

        You can append new rows as a `pandas.DataFrame`,

        >>> import pandas as pd
        >>> engine = Animals()
        >>> crab_and_sponge = pd.DataFrame(
        ...     {
        ...         "index": ["crab", "sponge"],
        ...         "water": [1, 1],
        ...         "flippers": [0, 0],
        ...     }
        ... ).set_index("index")
        >>> engine.append_rows(crab_and_sponge)
        >>> engine.index[-1]
        'sponge'
        >>> engine[-1, "water"]
        1

        or a `pandas.Series`

        >>> squid = pd.Series([0, 1], index=["water", "slow"], name="squid")
        >>> engine.append_rows(squid)
        >>> engine.index[-1]
        'squid'
        >>> engine[-1, "slow"]
        1

        or a dictionary of dictionaries

        >>> engine = Animals()
        >>> rows = {
        ...     "crab": {"water": 1, "flippers": 0},
        ...     "sponge": {"water": 1, "flippers": 0},
        ...     "squid": {"water": 1, "slow": 1},
        ... }
        >>> engine.append_rows(rows)
        >>> engine.index[-3:]
        ['crab', 'sponge', 'squid']
        >>> engine[-3:, "flippers"]  # doctest: +NORMALIZE_WHITESPACE
        shape: (3, 2)
        ┌────────┬──────────┐
        │ Index  ┆ flippers │
        │ ---    ┆ ---      │
        │ str    ┆ u8       │
        ╞════════╪══════════╡
        │ crab   ┆ 0        │
        │ sponge ┆ 0        │
        │ squid  ┆ null     │
        └────────┴──────────┘
        """
        if isinstance(rows, dict):
            for name, values in rows.items():
                row = pd.Series(values, name=name)
                self.engine.append_rows(row)
        else:
            self.engine.append_rows(rows)

    def append_columns(
        self,
        cols: Union[pd.DataFrame, pl.DataFrame],
        metadata: Optional[List[core.ColumnMetadata]] = None,
        cat_cutoff: int = 20,
        no_hypers: bool = False,
    ):
        """
        Append new columns to the Engine.

        Parameters
        ----------
        cols: polars.DataFrame, pandas.DataFrame
            The new column(s) to append to the ``Engine``. If ``cols`` is a
            polars DataFrame, cols must contain an ``ID`` column. Note that new
            indices will result in new rows
        col_metadata: dict[str, ColumnMetadata], Optional
            A map from column name to metadata. If None (default) metadata will
            be inferred from the data.
        cat_cutoff: int, optional
            The max value of an unsigned integer a column can have before it is
            inferred to be count type (default: 20). Used only if
            ``col_metadata`` is None.
        no_hypers: bool, optional
            If True, the prior will be fixed and hyper priors will be ignored.
            Used only if ``col_metadata`` is None.

        Examples
        --------
        Append a new continuous column

        >>> import numpy as np
        >>> import polars as pl
        >>> from lace.examples import Animals
        >>> engine = Animals()
        >>> engine.shape
        (50, 85)
        >>> column = pl.DataFrame([
        ...     pl.Series("index", engine.index),  # index
        ...     pl.Series("rand", np.random.randn(engine.shape[0])),
        ... ])
        >>> engine.append_columns(column)
        >>> engine.shape
        (50, 86)
        >>> engine.ftype("rand")
        'Continuous'

        Also works with pandas DataFrames

        >>> import pandas as pd
        >>> engine = Animals()
        >>> engine.shape
        (50, 85)
        >>> column = pd.DataFrame({
        ...     "rand": np.random.randn(engine.shape[0]),
        ... }, index=engine.index)
        >>> engine.append_columns(column)
        >>> engine.shape
        (50, 86)
        >>> engine.ftype("rand")
        'Continuous'

        You can append multiple columns

        >>> engine = Animals()
        >>> engine.shape
        (50, 85)
        >>> columns = pd.DataFrame({
        ...     "rand1": np.random.randn(engine.shape[0]),
        ...     "rand2": np.random.randn(engine.shape[0]),
        ... }, index=engine.index)
        >>> engine.append_columns(columns)
        >>> engine.shape
        (50, 87)
        >>> engine.ftype("rand1")
        'Continuous'
        >>> engine.ftype("rand2")
        'Continuous'

        And you can append partially filled columns

        >>> engine = Animals()
        >>> engine.shape
        (50, 85)
        >>> columns = pd.DataFrame({
        ...     "values": [0.0, 1.0, 2.0],
        ... }, index=[engine.index[0], engine.index[2], engine.index[5]])
        >>> engine.append_columns(columns)
        >>> engine[:7, "values"]  # doctest: +NORMALIZE_WHITESPACE
        shape: (7, 2)
        ┌──────────────┬────────┐
        │ Index        ┆ values │
        │ ---          ┆ ---    │
        │ str          ┆ f64    │
        ╞══════════════╪════════╡
        │ antelope     ┆ 0.0    │
        │ grizzly+bear ┆ null   │
        │ killer+whale ┆ 1.0    │
        │ beaver       ┆ null   │
        │ dalmatian    ┆ null   │
        │ persian+cat  ┆ 2.0    │
        │ horse        ┆ null   │
        └──────────────┴────────┘

        We can append categorical columns as well. Sometimes you will need to
        define the metadata manually. In this case, there are more possible
        categories that categories observed in the data.

        >>> from lace import ColumnMetadata, CategoricalPrior, ValueMap
        >>> engine = Animals()
        >>> engine.shape
        (50, 85)
        >>> columns = pd.DataFrame({
        ...     "fav_color": ["Yellow", "Yellow", "Blue", "Sparkles"],
        ... }, index=engine.index[:4])
        >>> metadata = [
        ...     ColumnMetadata.categorical(
        ...         "fav_color",
        ...         4,
        ...         prior=CategoricalPrior(4),
        ...         value_map=ValueMap.string(["Blue", "Yellow", "Sparkles", "Green"])
        ...     ),
        ... ]
        >>> engine.append_columns(columns, metadata)
        >>> engine[:5, "fav_color"]  # doctest: +NORMALIZE_WHITESPACE
        shape: (5, 2)
        ┌──────────────┬───────────┐
        │ Index        ┆ fav_color │
        │ ---          ┆ ---       │
        │ str          ┆ str       │
        ╞══════════════╪═══════════╡
        │ antelope     ┆ Yellow    │
        │ grizzly+bear ┆ Yellow    │
        │ killer+whale ┆ Blue      │
        │ beaver       ┆ Sparkles  │
        │ dalmatian    ┆ null      │
        └──────────────┴───────────┘

        And count columns

        >>> engine = Animals()
        >>> engine.shape
        (50, 85)
        >>> columns = pd.DataFrame({
        ...     "times_watched_the_fifth_element": list(range(5)) * 10,
        ... }, index=engine.index)
        >>> engine.append_columns(columns, cat_cutoff=3)
        >>> engine[:8, "times_watched_the_fifth_element"]  # doctest: +NORMALIZE_WHITESPACE
        shape: (8, 2)
        ┌─────────────────┬─────────────────────────────────┐
        │ Index           ┆ times_watched_the_fifth_element │
        │ ---             ┆ ---                             │
        │ str             ┆ u32                             │
        ╞═════════════════╪═════════════════════════════════╡
        │ antelope        ┆ 0                               │
        │ grizzly+bear    ┆ 1                               │
        │ killer+whale    ┆ 2                               │
        │ beaver          ┆ 3                               │
        │ dalmatian       ┆ 4                               │
        │ persian+cat     ┆ 0                               │
        │ horse           ┆ 1                               │
        │ german+shepherd ┆ 2                               │
        └─────────────────┴─────────────────────────────────┘
        """
        if metadata is None:
            metadata = utils.infer_column_metadata(
                cols, cat_cutoff=cat_cutoff, no_hypers=no_hypers
            )

        self.engine.append_columns(cols, metadata)

    def del_column(self, col: Union[str, int]) -> None:
        """
        Delete a given column.

        Parameters
        ----------
        col: str or int
            The index of the column to delete

        Raises
        ------
        IndexError
            The requested column index does not exist or is out of bounds

        Examples
        --------
        Delete columns by integer or string index

        >>> from lace.examples import Animals
        >>> engine = Animals()
        >>> engine.shape
        (50, 85)
        >>> engine.del_column("swims")
        >>> engine.shape
        (50, 84)
        >>> engine.del_column(0)
        >>> engine.shape
        (50, 83)
        >>> engine.del_column(82)
        >>> engine.shape
        (50, 82)
        """
        self.engine.del_column(col)

    def update(
        self,
        n_iters: int,
        *,
        timeout: Optional[int] = None,
        checkpoint: Optional[int] = None,
        transitions: Optional[core.StateTransition] = None,
        save_path: Optional[Union[str, bytes, PathLike]] = None,
        quiet: bool = False,
    ):
        """
        Update the Engine by advancing the Markov chains.

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
        transitions: List[StateTransition], optional
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
        ...     100,
        ...     timeout=30,
        ...     transitions=[
        ...         StateTransition.row_assignment(RowKernel.slice()),
        ...         StateTransition.view_alphas(),
        ...     ],
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
        Estimate the entropy or joint entropy of one or more features.

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
        >>> animals.entropy(["slow"])
        0.6755931727528786
        >>> animals.entropy(["water"])
        0.49836129824622094

        Joint entropy

        >>> animals.entropy(["swims", "fast"])
        0.9552642751735604

        We can use entropies to compute mutual information, I(X, Y) = H(X) +
        H(Y) - H(X, Y).

        For example, there is not a lot of shared information between whether an
        animals swims and whether it is fast. These features are not predictive
        of each other.

        >>> h_swims = animals.entropy(["swims"])
        >>> h_fast = animals.entropy(["fast"])
        >>> h_swims_and_fast = animals.entropy(["swims", "fast"])
        >>> h_swims + h_fast - h_swims_and_fast
        3.510013543328583e-05

        But swimming and having flippers are mutually predictive, so we should
        see more mutual information.

        >>> h_flippers = animals.entropy(["flippers"])
        >>> h_swims_and_flippers = animals.entropy(["swims", "flippers"])
        >>> h_swims + h_flippers - h_swims_and_flippers
        0.19361180218629537
        """
        return self.engine.entropy(cols, n_mc_samples)

    def logp(
        self,
        values,
        given=None,
        *,
        state_ixs: Optional[List[int]] = None,
        scaled: bool = False,
    ) -> Union[None, float, pl.Series]:
        r"""
        Compute the log likelihood.

        This function computes ``log p(values)`` or ``log p(values|given)``.

        Parameters
        ----------
        values: polars or pandas DataFrame or Series
            The values over which to compute the log likelihood. Each row of the
            DataFrame, or each entry of the Series, is an observation. Column
            names (or the Series name) should correspond to names of features in
            the table.
        given: Dict[index, value], optional
            A dictionary mapping column indices/name to values, which specifies
            conditions on the observations.
        state_ixs: List[int], optional
            An optional list specifying which states should be used in the
            likelihood computation. If `None` (default), use all states.
        scaled: bool, optional
            If `True` the components of the likelihoods will be scaled so that
            each dimension (feature) contributes a likelihood in [0, 1], thus
            the scaled log likelihood will not be as prone to being dominated
            by any one feature.

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
        >>> class_of_orbit = pl.Series("Class_of_Orbit", ["LEO", "MEO", "GEO"])
        >>> engine.logp(class_of_orbit).exp()  # doctest: +NORMALIZE_WHITESPACE
        shape: (3,)
        Series: 'logp' [f64]
        [
            0.523575
            0.06601
            0.380453
        ]

        Conditioning using ``given``

        >>> engine.logp(
        ...     class_of_orbit,
        ...     given={"Period_minutes": 1436.0},
        ... ).exp()  # doctest: +NORMALIZE_WHITESPACE
        shape: (3,)
        Series: 'logp' [f64]
        [
            0.000349
            0.000756
            0.998017
        ]

        Ask about the likelihood of values belonging to multiple features

        >>> values = pl.DataFrame(
        ...     {
        ...         "Class_of_Orbit": ["LEO", "MEO", "GEO"],
        ...         "Period_minutes": [70.0, 320.0, 1440.0],
        ...     }
        ... )
        >>> engine.logp(values).exp()  # doctest: +NORMALIZE_WHITESPACE
        shape: (3,)
        Series: 'logp' [f64]
        [
            0.000306
            0.000008
            0.016546
        ]

        An example of the scaled variant:

        >>> engine.logp(
        ...     values,
        ...     scaled=True,
        ... ).exp()  # doctest: +NORMALIZE_WHITESPACE
        shape: (3,)
        Series: 'logp_scaled' [f64]
        [
            0.137554
            0.167357
            0.577699
        ]

        For columns which we explicitly model missing-not-at-random data, we can
        ask about the likelihood of missing values.

        >>> from math import exp
        >>> no_long_geo = pl.Series("longitude_radians_of_geo", [None])
        >>> exp(engine.logp(no_long_geo))
        0.631030460838865

        The probability of a value missing (not-at-random) changes depending on
        the conditions.

        >>> exp(engine.logp(no_long_geo, given={"Class_of_Orbit": "GEO"}))
        0.048855132811982976

        And we can condition on missingness

        >>> engine.logp(
        ...     class_of_orbit,
        ...     given={"longitude_radians_of_geo": None},
        ... ).exp()  # doctest: +NORMALIZE_WHITESPACE
        shape: (3,)
        Series: 'logp' [f64]
        [
            0.827158
            0.099435
            0.029606
        ]

        Plot the marginal distribution of `Period_minutes` for each state

        >>> import numpy as np
        >>> import plotly.graph_objects as go
        >>> period = pl.Series('Period_minutes', np.linspace(0, 1500, 500))
        >>> fig = go.Figure()
        >>> for i in range(engine.n_states):
        ...     p = engine.logp(period, state_ixs=[i]).exp()
        ...     fig = fig.add_trace(go.Scatter(
        ...         x=period,
        ...         y=p,
        ...         name=f'state {i}',
        ...         hoverinfo='text+name',
        ...     ))
        >>> fig.update_layout(
        ...         xaxis_title='Period_minutes',
        ...         yaxis_title='f(Period)',
        ...     ) \
        ...     .show()
        """
        srs = (
            self.engine.logp_scaled(values, given, state_ixs)
            if scaled
            else self.engine.logp(values, given, state_ixs)
        )

        return utils.return_srs(srs)

    def inconsistency(self, values, given=None):
        """
        Compute inconsistency.

        Parameters
        ----------
        values: polars or pandas DataFrame or Series
            The values over which to compute the inconsistency. Each row of the
            DataFrame, or each entry of the Series, is an observation. Column
            names (or the Series name) should correspond to names of features in
            the table.
        given: Dict[index, value], optional
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
        >>> index = animals.df["id"]
        >>> inconsistency = animals.inconsistency(animals.df.drop("id"))
        >>> pl.DataFrame({"index": index, "inconsistency": inconsistency}).sort(
        ...     "inconsistency", descending=True
        ... )  # doctest: +NORMALIZE_WHITESPACE
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
        │ …              ┆ …             │
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
        ...     period = given.pop("Period_minutes", None)
        ...
        ...     if period is None:
        ...         continue
        ...
        ...     ix = given.pop("ID")
        ...     ic = engine.inconsistency(
        ...         pl.Series("Period_minutes", [period]),
        ...         given,
        ...     )
        ...
        ...     data.append(
        ...         {
        ...             "index": ix,
        ...             "inconsistency": ic,
        ...             "Period_minutes": period,
        ...         }
        ...     )
        ...
        >>> pl.DataFrame(data).sort(
        ...     "inconsistency", descending=True
        ... )  # doctest: +NORMALIZE_WHITESPACE
        shape: (1_162, 3)
        ┌───────────────────────────────────┬───────────────┬────────────────┐
        │ index                             ┆ inconsistency ┆ Period_minutes │
        │ ---                               ┆ ---           ┆ ---            │
        │ str                               ┆ f64           ┆ f64            │
        ╞═══════════════════════════════════╪═══════════════╪════════════════╡
        │ Intelsat 903                      ┆ 1.973348      ┆ 1436.16        │
        │ TianLian 2 (TL-1-02, CTDRS)       ┆ 1.3645        ┆ 1436.1         │
        │ QZS-1 (Quazi-Zenith Satellite Sy… ┆ 1.364247      ┆ 1436.0         │
        │ Compass G-8 (Beidou IGSO-3)       ┆ 1.364093      ┆ 1435.93        │
        │ …                                 ┆ …             ┆ …              │
        │ Navstar GPS II-24 (Navstar SVN 3… ┆ 0.646141      ┆ 716.69         │
        │ Navstar GPS IIR-10 (Navstar SVN … ┆ 0.646027      ┆ 716.47         │
        │ Navstar GPS IIR-M-6 (Navstar SVN… ┆ 0.645991      ┆ 716.4          │
        │ BSAT-3B                           ┆ 0.625282      ┆ 1365.61        │
        └───────────────────────────────────┴───────────────┴────────────────┘

        It looks like Intelsat 903 is the most inconsistent by a good amount.
        Let's take a look at it's data and see why its orbital period (very
        standard for a geosynchronos satellites) isn't consistent with the model.

        >>> cols = [
        ...     "Period_minutes",
        ...     "Class_of_Orbit",
        ...     "Perigee_km",
        ...     "Apogee_km",
        ...     "Eccentricity",
        ... ]
        >>> engine.df.filter(pl.col("ID") == "Intelsat 903")[cols].melt()
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

    def surprisal(
        self, col: Union[int, str], *, rows=None, values=None, state_ixs=None
    ):
        r"""
        Compute the surprisal of a values in specific cells.

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
            Proposed values for each cell. Must have an entry for each entry
            in `rows`. If `None`, the existing values are used.
        state_ixs: List[int], optional
            An optional list specifying which states should be used in the
            surprisal computation. If `None` (default), use all states.

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
        >>> engine.surprisal("Expected_Lifetime").sort(
        ...     "surprisal", descending=True
        ... ).head(5)
        shape: (5, 3)
        ┌───────────────────────────────────┬───────────────────┬───────────┐
        │ index                             ┆ Expected_Lifetime ┆ surprisal │
        │ ---                               ┆ ---               ┆ ---       │
        │ str                               ┆ f64               ┆ f64       │
        ╞═══════════════════════════════════╪═══════════════════╪═══════════╡
        │ International Space Station (ISS… ┆ 30.0              ┆ 7.02499   │
        │ Landsat 7                         ┆ 15.0              ┆ 4.869031  │
        │ Milstar DFS-5 (USA 164, Milstar … ┆ 0.0               ┆ 4.74869   │
        │ Optus B3                          ┆ 0.5               ┆ 4.653549  │
        │ SDS III-3 (Satellite Data System… ┆ 0.5               ┆ 4.558333  │
        └───────────────────────────────────┴───────────────────┴───────────┘


        Compute the surprisal for specific cells

        >>> engine.surprisal(
        ...     "Expected_Lifetime", rows=["Landsat 7", "Intelsat 701"]
        ... )
        shape: (2, 3)
        ┌──────────────┬───────────────────┬───────────┐
        │ index        ┆ Expected_Lifetime ┆ surprisal │
        │ ---          ┆ ---               ┆ ---       │
        │ str          ┆ f64               ┆ f64       │
        ╞══════════════╪═══════════════════╪═══════════╡
        │ Landsat 7    ┆ 15.0              ┆ 4.869031  │
        │ Intelsat 701 ┆ 0.5               ┆ 4.533067  │
        └──────────────┴───────────────────┴───────────┘

        Compute the surprisal of specific values in specific cells

        >>> engine.surprisal(
        ...     "Expected_Lifetime",
        ...     rows=["Landsat 7", "Intelsat 701"],
        ...     values=[10.0, 10.0],
        ... )
        shape: (2, 3)
        ┌──────────────┬───────────────────┬───────────┐
        │ index        ┆ Expected_Lifetime ┆ surprisal │
        │ ---          ┆ ---               ┆ ---       │
        │ str          ┆ f64               ┆ f64       │
        ╞══════════════╪═══════════════════╪═══════════╡
        │ Landsat 7    ┆ 10.0              ┆ 3.037384  │
        │ Intelsat 701 ┆ 10.0              ┆ 2.559729  │
        └──────────────┴───────────────────┴───────────┘

        Compute the surprisal of multiple values in a single cell

        >>> engine.surprisal(
        ...     "Expected_Lifetime",
        ...     rows=["Landsat 7"],
        ...     values=[0.5, 1.0, 5.0, 10.0],
        ... )  # doctest: +NORMALIZE_WHITESPACE
        shape: (4,)
        Series: 'surprisal' [f64]
        [
            3.126282
            2.938583
            2.24969
            3.037384
        ]

        Surprisal will be different under different_states

        >>> engine.surprisal(
        ...     "Expected_Lifetime",
        ...     rows=["Landsat 7", "Intelsat 701"],
        ...     values=[10.0, 10.0],
        ...     state_ixs=[0, 1],
        ... )
        shape: (2, 3)
        ┌──────────────┬───────────────────┬───────────┐
        │ index        ┆ Expected_Lifetime ┆ surprisal │
        │ ---          ┆ ---               ┆ ---       │
        │ str          ┆ f64               ┆ f64       │
        ╞══════════════╪═══════════════════╪═══════════╡
        │ Landsat 7    ┆ 10.0              ┆ 2.743636  │
        │ Intelsat 701 ┆ 10.0              ┆ 2.587096  │
        └──────────────┴───────────────────┴───────────┘
        """
        out = self.engine.surprisal(
            col, rows=rows, values=values, state_ixs=state_ixs
        )

        if out.shape[1] == 1:
            return out["surprisal"]
        else:
            return out

    def simulate(
        self, cols, given=None, n: int = 1, include_given: bool = False
    ):
        """
        Simulate data from a conditional distribution.

        Parameters
        ----------
        cols: List[column index]
            A list of target columns to simulate
        given: Dict[column index, value], optional
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
        >>> engine.simulate(["Class_of_Orbit", "Period_minutes"], n=5)
        shape: (5, 2)
        ┌────────────────┬────────────────┐
        │ Class_of_Orbit ┆ Period_minutes │
        │ ---            ┆ ---            │
        │ str            ┆ f64            │
        ╞════════════════╪════════════════╡
        │ MEO            ┆ 2807.568333    │
        │ GEO            ┆ 1421.333515    │
        │ LEO            ┆ 92.435621      │
        │ GEO            ┆ 1435.7067      │
        │ LEO            ┆ 84.896787      │
        └────────────────┴────────────────┘

        Simulate a pair of columns conditioned on another

        >>> engine.simulate(
        ...     ["Class_of_Orbit", "Period_minutes"],
        ...     given={"Purpose": "Communications"},
        ...     n=5,
        ... )
        shape: (5, 2)
        ┌────────────────┬────────────────┐
        │ Class_of_Orbit ┆ Period_minutes │
        │ ---            ┆ ---            │
        │ str            ┆ f64            │
        ╞════════════════╪════════════════╡
        │ GEO            ┆ 1439.041087    │
        │ GEO            ┆ 1426.020318    │
        │ GEO            ┆ 1430.553113    │
        │ GEO            ┆ 1451.192889    │
        │ GEO            ┆ 1431.855712    │
        └────────────────┴────────────────┘

        Simulate missing values for columns that are missing not-at-random

        >>> engine.simulate(["longitude_radians_of_geo"], n=5)
        shape: (5, 1)
        ┌──────────────────────────┐
        │ longitude_radians_of_geo │
        │ ---                      │
        │ f64                      │
        ╞══════════════════════════╡
        │ null                     │
        │ null                     │
        │ null                     │
        │ null                     │
        │ null                     │
        └──────────────────────────┘
        >>> engine.simulate(
        ...     ["longitude_radians_of_geo"],
        ...     given={"Class_of_Orbit": "GEO"},
        ...     n=5,
        ... )
        shape: (5, 1)
        ┌──────────────────────────┐
        │ longitude_radians_of_geo │
        │ ---                      │
        │ f64                      │
        ╞══════════════════════════╡
        │ 0.396442                 │
        │ 0.794023                 │
        │ 0.643669                 │
        │ -0.005531                │
        │ 1.827976                 │
        └──────────────────────────┘

        If we simulate using ``given`` conditions, we can include the
        conditions in the output using ``include_given=True``.

        >>> engine.simulate(
        ...     ["Period_minutes"],
        ...     given={"Purpose": "Communications", "Class_of_Orbit": "GEO"},
        ...     n=5,
        ...     include_given=True,
        ... )
        shape: (5, 3)
        ┌────────────────┬────────────────┬────────────────┐
        │ Period_minutes ┆ Purpose        ┆ Class_of_Orbit │
        │ ---            ┆ ---            ┆ ---            │
        │ f64            ┆ str            ┆ str            │
        ╞════════════════╪════════════════╪════════════════╡
        │ 1436.038447    ┆ Communications ┆ GEO            │
        │ 1447.908161    ┆ Communications ┆ GEO            │
        │ 1452.635331    ┆ Communications ┆ GEO            │
        │ 1443.983013    ┆ Communications ┆ GEO            │
        │ 1437.544045    ┆ Communications ┆ GEO            │
        └────────────────┴────────────────┴────────────────┘
        """
        df = self.engine.simulate(cols, given=given, n=n)

        if include_given and given is not None:
            for k, v in given.items():
                col = pl.Series(k, [v] * n)
                df = df.with_columns(col)

        return df

    def draw(self, row: Union[int, str], col: Union[int, str], n: int = 1):
        """
        Draw data from the distribution of a specific cell in the table.

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
        >>> engine.draw(
        ...     "Landsat 7", "Period_minutes", n=5
        ... )  # doctest: +NORMALIZE_WHITESPACE
        shape: (5,)
        Series: 'Period_minutes' [f64]
        [
            110.076567
            108.096406
            102.34334
            90.175641
            94.512276
        ]
        """
        srs = self.engine.draw(row, col, n)
        return utils.return_srs(srs)

    def predict(
        self,
        target: Union[str, int],
        given: Optional[Dict[Union[str, int], object]] = None,
        state_ixs: Optional[List[int]] = None,
        with_uncertainty: bool = True,
    ):
        """
        Predict a single target from a conditional distribution.

        Parameters
        ----------
        target: column index
            The column to predict
        given: Dict[column index, value], optional
            Column -> Value dictionary describing observations. Note that
            columns can either be indices (int) or names (str)
        state_ixs: List[int], optional
            An optional list specifying which states should be used in the
            prediction. If `None` (default), use all states.
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
        >>> animals.predict("swims")
        (0, 0.008287057807910558)

        Predict whether an animal swims given that it has flippers

        >>> animals.predict("swims", given={"flippers": 1})
        (1, 0.05008037071634858)

        Let's confuse lace and see what happens to its uncertainty. Let's
        predict whether an non-water animal with flippers swims

        >>> animals.predict("swims", given={"flippers": 1, "water": 0})
        (0, 0.32863593091906085)

        If you want to save time and you do not care about quantifying your
        epistemic uncertainty, you don't have to compute uncertainty.

        >>> animals.predict("swims", with_uncertainty=False)
        0
        """
        return self.engine.predict(target, given, state_ixs, with_uncertainty)

    def impute(
        self,
        col: Union[str, int],
        rows: Optional[List[Union[str, int]]] = None,
        unc_type: Optional[str] = "js_divergence",
    ):
        r"""
        Impute (predict) the value of a cell(s) in the lace table.

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
              \frac{1}{S^2 - S} \sum_{i=1}^S \sum{j \in \{1,..,S\} \setminus i} KL(X_i | X_j)

        Parameters
        ----------
        col: column index
            The column index
        rows: List[row index], optional
            Optional row indices to impute. If ``None`` (default), all the rows
            with missing values will be imputed
        unc_type: str, optional
            The type of uncertainty to compute. If ``None``, uncertainty will
            not be computed. Acceptable values are:

            - 'js_divergence' (default): The Jensen-Shannon divergence between the
              imputation distributions in each state.
            - 'pairwise_kl': The mean pairwise Kullback-Leibler divergence
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
        >>> engine.impute("Purpose")
        shape: (0, 2)
        ┌───────┬─────────┐
        │ index ┆ Purpose │
        │ ---   ┆ ---     │
        │ str   ┆ str     │
        ╞═══════╪═════════╡
        └───────┴─────────┘

        Let's choose a column that actually has missing values

        >>> engine.impute("Type_of_Orbit")  # doctest: +NORMALIZE_WHITESPACE
        shape: (645, 3)
        ┌───────────────────────────────────┬─────────────────┬─────────────┐
        │ index                             ┆ Type_of_Orbit   ┆ uncertainty │
        │ ---                               ┆ ---             ┆ ---         │
        │ str                               ┆ str             ┆ f64         │
        ╞═══════════════════════════════════╪═════════════════╪═════════════╡
        │ AAUSat-3                          ┆ Sun-Synchronous ┆ 0.102337    │
        │ ABS-1 (LMI-1, Lockheed Martin-In… ┆ Sun-Synchronous ┆ 0.445436    │
        │ ABS-1A (Koreasat 2, Mugunghwa 2,… ┆ Sun-Synchronous ┆ 0.544273    │
        │ ABS-2i (MBSat, Mobile Broadcasti… ┆ Sun-Synchronous ┆ 0.445436    │
        │ …                                 ┆ …               ┆ …           │
        │ Zhongxing 20A                     ┆ Sun-Synchronous ┆ 0.445436    │
        │ Zhongxing 22A (Chinastar 22A)     ┆ Sun-Synchronous ┆ 0.522895    │
        │ Zhongxing 2A (Chinasat 2A)        ┆ Sun-Synchronous ┆ 0.445436    │
        │ Zhongxing 9 (Chinasat 9, Chinast… ┆ Sun-Synchronous ┆ 0.445436    │
        └───────────────────────────────────┴─────────────────┴─────────────┘

        Impute a defined set of rows

        >>> engine.impute("Purpose", rows=["AAUSat-3", "Zhongxing 20A"])
        shape: (2, 3)
        ┌───────────────┬────────────────────────┬─────────────┐
        │ index         ┆ Purpose                ┆ uncertainty │
        │ ---           ┆ ---                    ┆ ---         │
        │ str           ┆ str                    ┆ f64         │
        ╞═══════════════╪════════════════════════╪═════════════╡
        │ AAUSat-3      ┆ Technology Development ┆ 0.1918      │
        │ Zhongxing 20A ┆ Communications         ┆ 0.191064    │
        └───────────────┴────────────────────────┴─────────────┘

        Uncertainty is optional

        >>> engine.impute("Type_of_Orbit", unc_type=None)  # doctest: +NORMALIZE_WHITESPACE
        shape: (645, 2)
        ┌───────────────────────────────────┬─────────────────┐
        │ index                             ┆ Type_of_Orbit   │
        │ ---                               ┆ ---             │
        │ str                               ┆ str             │
        ╞═══════════════════════════════════╪═════════════════╡
        │ AAUSat-3                          ┆ Sun-Synchronous │
        │ ABS-1 (LMI-1, Lockheed Martin-In… ┆ Sun-Synchronous │
        │ ABS-1A (Koreasat 2, Mugunghwa 2,… ┆ Sun-Synchronous │
        │ ABS-2i (MBSat, Mobile Broadcasti… ┆ Sun-Synchronous │
        │ …                                 ┆ …               │
        │ Zhongxing 20A                     ┆ Sun-Synchronous │
        │ Zhongxing 22A (Chinastar 22A)     ┆ Sun-Synchronous │
        │ Zhongxing 2A (Chinasat 2A)        ┆ Sun-Synchronous │
        │ Zhongxing 9 (Chinasat 9, Chinast… ┆ Sun-Synchronous │
        └───────────────────────────────────┴─────────────────┘
        """
        return self.engine.impute(col, rows, unc_type)

    def depprob(self, col_pairs: list):
        """
        Compute the dependence probability between pairs of columns.

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
        >>> engine.depprob([("swims", "flippers")])
        1.0

        Multiple pairs as inputs gets you a polars ``Series``

        >>> engine.depprob(
        ...     [
        ...         ("swims", "flippers"),
        ...         ("fast", "tail"),
        ...     ]
        ... )  # doctest: +NORMALIZE_WHITESPACE
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
        """
        Compute the mutual information between pairs of columns.

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
        >>> engine.mi([("swims", "flippers")])
        0.27197816458827445

        You can select different normalizations of mutual information

        >>> engine.mi([("swims", "flippers")], mi_type="unnormed")
        0.19361180218629537

        Multiple pairs as inputs gets you a polars ``Series``

        >>> engine.mi(
        ...     [
        ...         ("swims", "flippers"),
        ...         ("fast", "tail"),
        ...     ]
        ... )  # doctest: +NORMALIZE_WHITESPACE
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
        """
        Compute the row similarity between pairs of rows.

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
        row_pairs: List[(row index, row index)]
            A list of row pairs for which to compute row similarity
        wrt: List[column index], optional
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
        >>> animals.rowsim([("beaver", "polar+bear")])
        0.6059523809523808

        What about if we weight similarity by columns and not the standard
        views?

        >>> animals.rowsim([("beaver", "polar+bear")], col_weighted=True)
        0.5698529411764706

        Not much change. How similar are they with respect to how we model their
        swimming?

        >>> animals.rowsim([("beaver", "polar+bear")], wrt=["swims"])
        0.875

        Very similar. But will all animals that swim be highly similar with
        respect to their swimming?

        >>> animals.rowsim([("otter", "polar+bear")], wrt=["swims"])
        0.375

        Lace predicts an otter's swimming for different reasons than a polar
        bear's.

        What is a Chihuahua more similar to, a wolf or a rat?

        >>> from lace.examples import Animals
        >>> engine = Animals()
        >>> engine.rowsim(
        ...     [
        ...         ("chihuahua", "wolf"),
        ...         ("chihuahua", "rat"),
        ...     ]
        ... )  # doctest: +NORMALIZE_WHITESPACE
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
        """Compute the novelty of a row."""
        return self.engine.novelty(row, wrt)

    def pairwise_fn(self, fn_name, indices: Optional[list] = None, **kwargs):
        """
        Compute a function for a set of pairs of rows or columns.

        Parameters
        ----------
        fn_name: str
            The name of the function: 'rowsim', 'mi', or 'depprob'
        indices: List[index], optional
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
        ...     "rowsim",
        ...     indices=["wolf", "rat", "otter"],
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
        │ …     ┆ …     ┆ …        │
        │ rat   ┆ otter ┆ 0.613095 │
        │ otter ┆ wolf  ┆ 0.492262 │
        │ otter ┆ rat   ┆ 0.613095 │
        │ otter ┆ otter ┆ 1.0      │
        └───────┴───────┴──────────┘

        Extra keyword arguments are passed to the parent function.

        >>> engine.pairwise_fn(
        ...     "rowsim",
        ...     indices=["wolf", "rat", "otter"],
        ...     col_weighted=True,
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
        │ …     ┆ …     ┆ …        │
        │ rat   ┆ otter ┆ 0.491176 │
        │ otter ┆ wolf  ┆ 0.302206 │
        │ otter ┆ rat   ┆ 0.491176 │
        │ otter ┆ otter ┆ 1.0      │
        └───────┴───────┴──────────┘

        If you do not provide indices, the function is computed for the product
        of all indices.

        >>> engine.pairwise_fn("rowsim")
        shape: (2_500, 3)
        ┌──────────┬──────────────┬──────────┐
        │ A        ┆ B            ┆ rowsim   │
        │ ---      ┆ ---          ┆ ---      │
        │ str      ┆ str          ┆ f64      │
        ╞══════════╪══════════════╪══════════╡
        │ antelope ┆ antelope     ┆ 1.0      │
        │ antelope ┆ grizzly+bear ┆ 0.464137 │
        │ antelope ┆ killer+whale ┆ 0.479613 │
        │ antelope ┆ beaver       ┆ 0.438467 │
        │ …        ┆ …            ┆ …        │
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
        fn_kwargs=None,
        **kwargs,
    ) -> ClusterMap:
        """
        Generate a clustermap of a pairwise function.

        Parameters
        ----------
        fn_name: str
            The name of the function: 'rowsim', 'mi', or 'depprob'
        indices: List[index], optional
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
        ...     "depprob", zmin=0, zmax=1, color_continuous_scale="greys"
        ... ).figure.show()

        Use the ``fn_kwargs`` keyword argument to pass keyword arguments to the
        target function.

        >>> animals.clustermap(
        ...     "rowsim",
        ...     zmin=0,
        ...     zmax=1,
        ...     color_continuous_scale="greys",
        ...     fn_kwargs={"wrt": ["swims"]},
        ... ).figure.show()
        """
        if fn_kwargs is None:
            fn_kwargs = {}

        fn = self.pairwise_fn(fn_name, indices, **fn_kwargs)

        df = fn.pivot(values=fn_name, index="A", columns="B")
        df, linkage = utils.hcluster(df, method=linkage_method)

        if not no_plot:
            fig = px.imshow(
                df[:, 1:],
                labels={"x": "A", "y": "B", "color": fn_name},
                y=df["A"],
                **kwargs,
            )
            return ClusterMap(df, linkage, fig)
        else:
            return ClusterMap(df, linkage)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
