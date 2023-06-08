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
        Get a column metadata.

        Examples
        --------
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
        >>> codebook
        Codebook 'my_table'
          state_alpha_prior: G(a: 1, β: 1)
          view_alpha_prior: G(a: 1, β: 1)
          columns: 85
          rows: 50
        >>> codebook.rename("Dennis")
        Codebook 'Dennis'
          state_alpha_prior: G(a: 1, β: 1)
          view_alpha_prior: G(a: 1, β: 1)
          columns: 85
          rows: 50
        """
        codebook = copy.copy(self)
        codebook.codebook.rename(name)
        return codebook

    def set_state_alpha_prior(self, shape: float = 1.0, rate: float = 1.0):
        """
        Return a copy of the codebook with a new state CRP alpha prior.

        Parameters
        ----------
        shape: float, optional
            The shape of the Gamma distribution prior on alpha: a positive
            floating point value in (0, Inf). Default is 1.
        rate: float, optional
            The rate of the Gamma distribution prior on alpha: a positive
            floating point value in (0, Inf). Default is 1.

        Examples
        --------
        >>> from lace.examples import Animals
        >>> codebook = Animals().codebook
        >>> codebook
        Codebook 'my_table'
          state_alpha_prior: G(a: 1, β: 1)
          view_alpha_prior: G(a: 1, β: 1)
          columns: 85
          rows: 50
        >>> codebook.set_state_alpha_prior(2.0, 3.1)
        Codebook 'my_table'
          state_alpha_prior: G(a: 2, β: 3.1)
          view_alpha_prior: G(a: 1, β: 1)
          columns: 85
          rows: 50
        """
        codebook = copy.copy(self)
        codebook.codebook.set_state_alpha_prior(shape, rate)
        return codebook

    def set_view_alpha_prior(self, shape: float = 1.0, rate: float = 1.0):
        """
        Return a copy of the codebook with a new view CRP alpha prior.

        Parameters
        ----------
        shape: float, optional
            The shape of the Gamma distribution prior on alpha: a positive
            floating point value in (0, Inf). Default is 1.
        rate: float, optional
            The rate of the Gamma distribution prior on alpha: a positive
            floating point value in (0, Inf). Default is 1.

        Examples
        --------
        >>> from lace.examples import Animals
        >>> codebook = Animals().codebook
        >>> codebook
        Codebook 'my_table'
          state_alpha_prior: G(a: 1, β: 1)
          view_alpha_prior: G(a: 1, β: 1)
          columns: 85
          rows: 50
        >>> codebook.set_view_alpha_prior(2.0, 3.1)
        Codebook 'my_table'
          state_alpha_prior: G(a: 1, β: 1)
          view_alpha_prior: G(a: 2, β: 3.1)
          columns: 85
          rows: 50
        """
        codebook = copy.copy(self)
        codebook.codebook.set_view_alpha_prior(shape, rate)
        return codebook

    def append_column_metadata(self, col_metadata: List[_lc.ColumnMetadata]):
        codebook = copy.copy(self)
        codebook.codebook.append_col_metadata(col_metadata)
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

    def json(self) -> str:
        """Return the codebook as a JSON string."""
        return self.codebook.json()

    def __repr__(self):
        return self.codebook.__repr__()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
