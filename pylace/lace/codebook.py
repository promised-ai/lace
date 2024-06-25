import copy
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

import lace.core as _lc


class _ColumnMetadataIndexer:
    def __init__(self, codebook):
        self.codebook = codebook

    def __getitem__(self, name: str) -> _lc.ColumnMetadata:
        return self.codebook.column_metadata(name)

    def __setitem__(self, name: str, value: _lc.ColumnMetadata):
        if value.name != name:
            raise KeyError(
                f"column_metadata has name '{value.name}', which is invalid "
                f"for metadata at index '{name}'"
            )

        self.codebook.set_column_metadata(name, value)

    def remove(self, name: str) -> _lc.ColumnMetadata:
        """Remove and return the column metadata with ``name``."""
        return self.codebook.remove_column_metadata(name)

    def extend(self, column_metadatas: List[_lc.ColumnMetadata]):
        """Append a number of column metadatas to the end of codebook."""
        self.codebook.append_column_metadata(column_metadatas)

    def append(self, column_metadata: _lc.ColumnMetadata):
        """Add a column metadata to the end of the codebook."""
        self.extend([column_metadata])


class Codebook:
    """
    Stores metadata about the lace ``Engine``.

    The codebook stores information about the CRP priors on columns-to-views
    (``state_crp_alpha``) and rows-to-categories (``view_crp_alpha``), the row
    names, and how to model each column including the type, the prior, and the
    hyper prior.

    Attributes
    ----------
    codebook: core.Codebook
        A reference to the underlying rust data structure

    """

    codebook: _lc.Codebook

    def __init__(self, codebook: _lc.Codebook):
        self.codebook = codebook

    @classmethod
    def empty(cls, name: str):
        """Create an empty codebook with a given name."""
        codebook = _lc.Codebook(name)
        obj = cls(codebook)

        return obj

    @classmethod
    def from_df(
        cls,
        name: str,
        df: Union[pd.DataFrame, pl.DataFrame],
        cat_cutoff: int = 20,
        no_hypers: bool = False,
    ):
        """
        Infer a codebook from a DataFrame.

        Parameters
        ----------
        name: str
            The name of the engine
        df: pandas.DataFrame or polars.DataFrame
            The data that will be used to create the engine. Note that polars
            DataFrame must have an `ID` or `Index` column.
        cat_cutoff: int, optional
            The maximum value an unsigned integer may take on before its column
            is considered `Count` type. Default is 20.
        no_hypers: bool, optional
            If `True`, disable hyper prior inference. Priors will be derived
            from the data and will remain static.

        """

        if isinstance(df, pd.DataFrame):
            df = pl.DataFrame(df.reset_index())

        codebook = _lc.codebook_from_df(df, cat_cutoff, no_hypers)
        codebook.rename(name)
        obj = cls(codebook)

        return obj

    @property
    def column_metadata(self) -> _ColumnMetadataIndexer:
        """
        Get/set a column metadata.

        Examples
        --------
        Get the metadata for column

        >>> from lace.examples import Animals
        >>> codebook = Animals().codebook
        >>> codebook.column_metadata["swims"]
        {
          "name": "swims",
          "coltype": {
            "Categorical": {
              "k": 2,
              "hyper": {
                "pr_alpha": {
                  "shape": 1.0,
                  "scale": 1.0
                }
              },
              "value_map": {
                "u8": 2
              },
              "prior": null
            }
          },
          "notes": null,
          "missing_not_at_random": false
        }

        Set the metadata for column

        >>> from lace import CategoricalPrior, ColumnMetadata, ValueMap
        >>> swims_metadata = ColumnMetadata.categorical(
        ...     "swims",
        ...     3,
        ...     value_map=ValueMap.int(3),
        ... ).missing_not_at_random(True)
        >>> codebook.column_metadata["swims"] = swims_metadata
        >>> codebook.column_metadata["swims"]
        {
          "name": "swims",
          "coltype": {
            "Categorical": {
              "k": 3,
              "hyper": null,
              "value_map": {
                "u8": 3
              },
              "prior": null
            }
          },
          "notes": null,
          "missing_not_at_random": true
        }

        If you try to set the metadata with the wrong name, you will get a
        talking to.

        >>> swims_metadata = ColumnMetadata.categorical(
        ...     "not-swims",
        ...     3,
        ...     value_map=ValueMap.int(3),
        ... ).missing_not_at_random(True)
        >>> try:
        ...     codebook.column_metadata["swims"] = swims_metadata
        ... except KeyError as err:
        ...     assert "'not-swims', which is invalid" in str(err)
        ... else:
        ...     assert False

        You can also use ``append``, ``extend``, and ``remove``, just like you
        would with a list

        >>> codebook.column_names[-5:]
        ['smart', 'group', 'solitary', 'nestspot', 'domestic']
        >>> codebook.column_metadata.append(
        ...    ColumnMetadata.continuous("number-in-wild")
        ... )
        >>> codebook.column_names[-5:]
        ['group', 'solitary', 'nestspot', 'domestic', 'number-in-wild']

        Extend the column metadata:

        >>> codebook.column_metadata.extend([
        ...    ColumnMetadata.categorical("eats-trash", 2),
        ...    ColumnMetadata.categorical("scary", 2),
        ... ])
        >>> codebook.column_names[-5:]
        ['nestspot', 'domestic', 'number-in-wild', 'eats-trash', 'scary']

        Remove a column metadata:

        >>> codebook.column_metadata.remove('eats-trash')
        {
          "name": "eats-trash",
          "coltype": {
            "Categorical": {
              "k": 2,
              "hyper": null,
              "value_map": {
                "u8": 2
              },
              "prior": null
            }
          },
          "notes": null,
          "missing_not_at_random": false
        }
        >>> codebook.column_names[-5:]
        ['solitary', 'nestspot', 'domestic', 'number-in-wild', 'scary']

        """
        return _ColumnMetadataIndexer(self.codebook)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        A (n_rows, n_cols) tuple.

        Examples
        --------
        >>> from lace.examples import Animals
        >>> codebook = Animals().codebook
        >>> codebook.shape
        (50, 85)

        """
        return self.codebook.shape

    @property
    def row_names(self) -> List[str]:
        """
        Contains the name of each row.

        Examples
        --------
        >>> from lace.examples import Animals
        >>> codebook = Animals().codebook
        >>> codebook.row_names[:4]
        ['antelope', 'grizzly+bear', 'killer+whale', 'beaver']

        """
        return self.codebook.row_names

    @property
    def column_names(self) -> List[str]:
        """
        Contains the name of each column.

        Examples
        --------
        >>> from lace.examples import Animals
        >>> codebook = Animals().codebook
        >>> codebook.column_names[:4]
        ['black', 'white', 'blue', 'brown']

        """
        return self.codebook.column_names

    def rename(self, name: str):
        """
        Return a copy of the codebook with a new name.

        Parameters
        ----------
        name: str
            The new name

        Examples
        --------
        >>> from lace.examples import Animals
        >>> codebook = Animals().codebook
        >>> codebook  # doctest: +NORMALIZE_WHITESPACE
        Codebook 'my_table'
          state_prior_process: DP(α ~ G(α: 1, β: 1))
          view_prior_process: DP(α ~ G(α: 1, β: 1))
          columns: 85
          rows: 50
        >>> codebook.rename("Dennis")
        Codebook 'Dennis'
          state_prior_process: DP(α ~ G(α: 1, β: 1))
          view_prior_process: DP(α ~ G(α: 1, β: 1))
          columns: 85
          rows: 50

        """
        codebook = copy.copy(self)
        codebook.codebook.rename(name)
        return codebook

    def set_state_prior_process(self, prior_process: _lc.PriorProcess):
        """
        Return a copy of the codebook with a new state PriorProcess.

        Parameters
        ----------
        prior_process: core.PriorProcess

        Examples
        --------
        >>> from lace.examples import Animals
        >>> from lace import PriorProcess
        >>> codebook = Animals().codebook
        >>> codebook  # doctest: +NORMALIZE_WHITESPACE
        Codebook 'my_table'
          state_prior_process: DP(α ~ G(α: 1, β: 1))
          view_prior_process: DP(α ~ G(α: 1, β: 1))
          columns: 85
          rows: 50
        >>> process = PriorProcess.pitman_yor(1.0, 2.0, 0.5, 0.5)
        >>> codebook.set_state_prior_process(process)
        Codebook 'my_table'
          state_prior_process: PYP(α ~ G(α: 1, β: 2), d ~ Beta(α: 0.5, β: 0.5))
          view_prior_process: DP(α ~ G(α: 1, β: 1))
          columns: 85
          rows: 50

        """
        codebook = copy.copy(self)
        codebook.codebook.set_state_prior_process(prior_process)
        return codebook

    def set_view_prior_process(self, prior_process: _lc.PriorProcess):
        """
        Return a copy of the codebook with a new view PriorProcess.

        Parameters
        ----------
        prior_process: core.PriorProcess

        Examples
        --------
        >>> from lace.examples import Animals
        >>> from lace import PriorProcess
        >>> codebook = Animals().codebook
        >>> codebook  # doctest: +NORMALIZE_WHITESPACE
        Codebook 'my_table'
          state_prior_process: DP(α ~ G(α: 1, β: 1))
          view_prior_process: DP(α ~ G(α: 1, β: 1))
          columns: 85
          rows: 50
        >>> process = PriorProcess.pitman_yor(1.0, 2.0, 0.5, 0.5)
        >>> codebook.set_view_prior_process(process)
        Codebook 'my_table'
          state_prior_process: DP(α ~ G(α: 1, β: 1))
          view_prior_process: PYP(α ~ G(α: 1, β: 2), d ~ Beta(α: 0.5, β: 0.5))
          columns: 85
          rows: 50

        """
        codebook = copy.copy(self)
        codebook.codebook.set_view_prior_process(prior_process)
        return codebook

    def append_column_metadata(self, col_metadata: List[_lc.ColumnMetadata]):
        """Append new columns to the codebook."""
        codebook = copy.copy(self)
        codebook.codebook.append_column_metadata(col_metadata)
        return codebook

    def set_row_names(
        self, row_names: Union[List[str], pd.Series, pl.Series, np.ndarray]
    ):
        """
        Return a copy of the codebook with new row_names.

        Examples
        --------
        >>> from lace.examples import Animals
        >>> codebook = Animals().codebook
        >>> codebook.row_names[:2]
        ['antelope', 'grizzly+bear']
        >>> codebook.set_row_names(["A", "B"]).row_names
        ['A', 'B']

        >>> import pandas as pd
        >>> new_rows = pd.Series(["one", "two", "three"])
        >>> codebook.set_row_names(new_rows).row_names
        ['one', 'two', 'three']

        >>> import polars as pl
        >>> new_rows = pl.Series("rows", ["one-1", "two-2", "three-3"])
        >>> codebook.set_row_names(new_rows).row_names
        ['one-1', 'two-2', 'three-3']

        >>> import numpy as np
        >>> new_rows = np.array(["one", "two", "three-hahaha"])
        >>> codebook.set_row_names(new_rows).row_names
        ['one', 'two', 'three-hahaha']

        """
        if isinstance(row_names, (pl.Series, pd.Series)):
            row_names = row_names.to_list()
        elif isinstance(row_names, np.ndarray):
            row_names = row_names.tolist()

        codebook = copy.copy(self)
        codebook.codebook.set_row_names(row_names)

        return codebook

    def value_map(self, col: str):
        """
        Get the value map for a Categorical column if it exists.

        Parameters
        ----------
        col: str
            The column name

        Examples
        --------
        String value map

        >>> from lace.examples import Satellites
        >>> sats = Satellites()
        >>> vm = sats.codebook.value_map("Class_of_Orbit")
        >>> vm
        ValueMap (String) [ 'Elliptical' 'GEO' 'LEO' 'MEO' ]
        >>> [c for c in vm.values()]
        ['Elliptical', 'GEO', 'LEO', 'MEO']

        Integer value map

        >>> from lace.examples import Animals
        >>> animals = Animals()
        >>> vm = animals.codebook.value_map("swims")
        >>> vm
        ValueMap (UInt, k=2)
        >>> [c for c in vm.values()]
        [0, 1]

        """
        return self.column_metadata[col].value_map

    def json(self) -> str:
        """Return the codebook as a JSON string."""
        return self.codebook.json()

    def __repr__(self):
        return self.codebook.__repr__()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
