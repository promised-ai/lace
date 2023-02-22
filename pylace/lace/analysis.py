from typing import Any
import polars as pl
from tqdm import tqdm

from lace import Engine


def _held_out_compute(
    engine: Engine,
    kind: str,
    values,
    given: dict[str | int, Any],
):
    if kind == 'neglogp':
        return -engine.logp(values, given=given)
    elif kind == 'inconsistency':
        return engine.inconsistency(values, given=given)
    else:
        raise ValueError(f'Invalid computation `{kind}`')


def _held_out_base(
    engine: Engine,
    kind: str,
    values,
    given: dict[str | int, Any],
    quiet: bool = False,
) -> pl.DataFrame:
    if not quiet:
        n = len(given)
        total = n**2 / 2 - n
        pbar = tqdm(total=total)

    # logp = engine.logp(values, given=given)
    f = _held_out_compute(engine, kind, values, given)
    n = len(given)

    fs = [f]
    rm_keys = [None]
    keys_removed = [0]

    all_keys = sorted(list(given.keys()))
    rm_dict = dict()

    if not quiet:
        pbar.update(1)

    for i in range(n):
        argmin = 0
        for ix, key in enumerate(all_keys):
            val = given.pop(key)

            f = _held_out_compute(engine, kind, values, given)

            if ix == 0:
                f_opt = f
            else:
                if f < f_opt:
                    argmin = ix
                    f_opt = f

            given[key] = val

            if not quiet:
                pbar.update(1)

        keys_removed.append(i+1)

        fs.append(f_opt)
        rm_keys.append(all_keys[argmin])
        
        key = all_keys[argmin]
        del all_keys[argmin]
        rm_dict[key] = given.pop(key)

    if not quiet:
        pbar.close()

    given = rm_dict

    return pl.DataFrame([
        pl.Series('feature_rmed', rm_keys),
        pl.Series(kind, fs),
        pl.Series('keys_rmed', keys_removed),
    ])


def held_out_neglogp(
    engine: Engine,
    values,
    given: dict[str | int, Any],
    quiet: bool = False,
) -> pl.DataFrame:
    """
    Compute -logp for values while sequentially dropping given conditions

    Parameters
    ----------
    engine: Engine
        The Engine used to compute logp
    values: polars or pandas DataFrame or Series
        The values over which to compute the log likelihood. Each row of the
        DataFrame, or each entry of the Series, is an observation. Column
        names (or the Series name) should correspond to names of features in
        the table.
    given: dict[index, value], optional
        A dictionary mapping column indices/name to values, which specifies
        conditions on the observations.

    Returns
    -------
    polars.DataFrame
        A DataFrame with a 'feature' column and a '-logp' column.

    Examples
    --------

    >>> import polars as pl
    >>> from lace.examples import Satellites
    >>> from lace.analysis import held_out_neglogp
    >>> satellites = Satellites()
    >>> given = satellites \\
    ...     .df \\
    ...     .to_pandas() \\
    ...     .set_index('ID') \\
    ...     .loc['Intelsat 903', :] \\
    ...     .dropna() \\
    ...     .to_dict()
    >>> period = given.pop('Period_minutes')
    >>> held_out_neglogp(
    ...     satellites,
    ...     pl.Series('Period_minutes', [period]),
    ...     given,
    ...     quiet=True,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    shape: (19, 3)
    ┌─────────────────────┬──────────┬───────────┐
    │ feature_rmed        ┆ neglogp  ┆ keys_rmed │
    │ ---                 ┆ ---      ┆ ---       │
    │ str                 ┆ f64      ┆ i64       │
    ╞═════════════════════╪══════════╪═══════════╡
    │ null                ┆ 7.380664 ┆ 0         │
    │ Apogee_km           ┆ 3.904223 ┆ 1         │
    │ Eccentricity        ┆ 2.995854 ┆ 2         │
    │ Country_of_Operator ┆ 2.995854 ┆ 3         │
    │ ...                 ┆ ...      ┆ ...       │
    │ Expected_Lifetime   ┆ 2.995911 ┆ 15        │
    │ Users               ┆ 2.996213 ┆ 16        │
    │ Inclination_radians ┆ 3.00236  ┆ 17        │
    │ Perigee_km          ┆ 4.009643 ┆ 18        │
    └─────────────────────┴──────────┴───────────┘
    """
    res = _held_out_base(
        engine,
        'neglogp',
        values,
        given,
        quiet=quiet,
    )
    return res


def held_out_inconsistency(
    engine: Engine,
    values,
    given: dict[str | int, Any],
    quiet: bool = False,
) -> pl.DataFrame:
    """
    Compute inconsistency for values while sequentially dropping given conditions

    Parameters
    ----------
    engine: Engine
        The Engine used to compute inconsistency
    values: polars or pandas DataFrame or Series
        The values over which to compute the inconsistency. Each row of the
        DataFrame, or each entry of the Series, is an observation. Column
        names (or the Series name) should correspond to names of features in
        the table.
    given: dict[index, value], optional
        A dictionary mapping column indices/name to values, which specifies
        conditions on the observations.

    Returns
    -------
    polars.DataFrame
        A DataFrame with a 'feature' column and a '-logp' column.

    Examples
    --------

    >>> import polars as pl
    >>> from lace.examples import Satellites
    >>> from lace.analysis import held_out_inconsistency
    >>> satellites = Satellites()
    >>> given = satellites \\
    ...     .df \\
    ...     .to_pandas() \\
    ...     .set_index('ID') \\
    ...     .loc['Intelsat 903', :] \\
    ...     .dropna() \\
    ...     .to_dict()
    >>> period = given.pop('Period_minutes')
    >>> held_out_inconsistency(
    ...     satellites,
    ...     pl.Series('Period_minutes', [period]),
    ...     given,
    ...     quiet=True,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    shape: (19, 3)
    ┌─────────────────────┬───────────────┬───────────┐
    │ feature_rmed        ┆ inconsistency ┆ keys_rmed │
    │ ---                 ┆ ---           ┆ ---       │
    │ str                 ┆ f64           ┆ i64       │
    ╞═════════════════════╪═══════════════╪═══════════╡
    │ null                ┆ 1.840728      ┆ 0         │
    │ Apogee_km           ┆ 0.973708      ┆ 1         │
    │ Eccentricity        ┆ 0.747162      ┆ 2         │
    │ Country_of_Operator ┆ 0.747162      ┆ 3         │
    │ ...                 ┆ ...           ┆ ...       │
    │ Expected_Lifetime   ┆ 0.747176      ┆ 15        │
    │ Users               ┆ 0.747252      ┆ 16        │
    │ Inclination_radians ┆ 0.748785      ┆ 17        │
    │ Perigee_km          ┆ 1.0           ┆ 18        │
    └─────────────────────┴───────────────┴───────────┘
    """
    res = _held_out_base(
        engine,
        'inconsistency',
        values,
        given,
        quiet=quiet,
    )
    return res


if __name__ == "__main__":
    import doctest
    doctest.testmod()
