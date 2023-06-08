import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from lace import ColumnMetadata, core


class _ColumnMetadataIndexer:
    def __init__(self, codebook):
        self.codebook = codebook

    def __getitem__(self, name: str) -> ColumnMetadata:
        return self.codebook.column_metadata(name)


class Codebook:
    codebook: core.Codebook

    def __init__(self, name: str):
        self.codebook = core.Codebook(name)

    @classmethod
    def from_df(
        cls,
        name: str,
        df: Union[pd.DataFrame, pl.DataFrame],
        cat_cutoff: Optional[bool] = None,
        no_hypers: bool = False,
    ):
        obj = cls(name)

        if isinstance(df, pd.DataFrame):
            df = pl.DataFrame(df.reset_index())

        codebook = core.codebook_from_df(df, cat_cutoff, no_hypers)
        codebook.rename(name)
        obj.codebook = codebook

        return obj

    @property
    def column_metadata(self) -> _ColumnMetadataIndexer:
        return _ColumnMetadataIndexer(self.codebook)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.codebook.shape

    @property
    def row_names(self) -> List[str]:
        return self.codebook.row_names

    @property
    def column_names(self) -> List[str]:
        return self.codebook.column_names

    def rename(self, name: str):
        codebook = copy.copy(self)
        codebook.codebook.rename(name)
        return codebook

    def set_state_alpha_prior(self, shape: float = 1.0, rate: float = 1.0):
        codebook = copy.copy(self)
        codebook.codebook.set_state_alpha_prior(shape, rate)
        return codebook

    def set_view_alpha_prior(self, shape: float = 1.0, rate: float = 1.0):
        codebook = copy.copy(self)
        codebook.codebook.set_view_alpha_prior(shape, rate)
        return codebook

    def append_column_metadata(self, col_metadata: List[ColumnMetadata]):
        codebook = copy.copy(self)
        codebook.codebook.append_col_metadata(col_metadata)
        return codebook

    def set_row_names(
        self, row_names: Union[List[str], pd.Series, pl.Series, np.ndarray]
    ):
        if isinstance(row_names, (pl.Series, pd.Series)):
            row_names = row_names.to_list()
        elif isinstance(row_names, np.ndarray):
            row_names = row_names.tolist()

        codebook = copy.copy(self)
        codebook.codebook.set_row_names(row_names)

        return codebook

    def json(self) -> str:
        return self.codebook.json()

    def __repr__(self):
        return self.codebook.__repr__()
