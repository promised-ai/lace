"""Tools for analysis of CrossCat results in Lace."""


import enum
import itertools as it
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, Union

import polars as pl
from tqdm import tqdm

if TYPE_CHECKING:
    from lace import Engine


class HoldOutSearchMethod(enum.Enum):
    """Method for hold out search."""

    Greedy = 0
    Enumerate = 1

    def __repr__(self) -> str:
        """Return the canonical string representation of the object."""
        if self == HoldOutSearchMethod.Greedy:
            return "greedy"
        elif self == HoldOutSearchMethod.Enumerate:
            return "enumerate"
        else:
            raise NotImplementedError


class HoldOutFunc(enum.Enum):
    """Hold out evaluation function."""

    NegLogp = 0
    Inconsistency = 1

    def __repr__(self) -> str:
        """Return the canonical string representation of the object."""
        if self == HoldOutFunc.NegLogp:
            return "-logp"
        elif self == HoldOutFunc.Inconsistency:
            return "inconsistency"
        else:
            raise NotImplementedError


def _held_out_compute(
    engine: "Engine",
    fn: HoldOutFunc,
    values,
    given: dict[Union[str, int], Any],
) -> Optional[float]:
    if fn == HoldOutFunc.NegLogp:
        logp = engine.logp(values, given=given)
        if logp is not None:
            return -logp
        else:
            return None
    elif fn == HoldOutFunc.Inconsistency:
        return engine.inconsistency(values, given=given)
    else:
        raise ValueError(f"Invalid computation `{fn}`")


def _held_out_inner_enum(
    engine: "Engine",
    fn: HoldOutFunc,
    n,
    values,
    given: dict[Union[str, int], Any],
    pbar: Optional[tqdm],
) -> tuple[float, set[str]]:
    all_keys = list(given.keys())
    all_keys.sort()

    f_opt = None
    argmin = None

    for keys in it.combinations(all_keys, n):
        temp = deepcopy(given)
        for key in keys:
            temp.pop(key)

        f = _held_out_compute(engine, fn, values, temp)
        if f is not None:
            if f_opt is None:
                f_opt = f
                argmin = temp
            else:
                if f < f_opt:
                    f_opt = f
                    argmin = temp

        if pbar is not None:
            pbar.update(1)

    return f_opt, sorted([k for k in given if k not in argmin])


def _held_out_inner_greedy(
    engine: "Engine",
    fn: HoldOutFunc,
    values,
    given: dict[Union[str, int], Any],
    pbar: Optional[tqdm],
) -> tuple[float, set[str]]:
    all_keys = list(given.keys())
    all_keys.sort()

    argmin = 0
    for ix, key in enumerate(all_keys):
        val = given.pop(key)

        f = _held_out_compute(engine, fn, values, given)

        if ix == 0:
            f_opt = f
        else:
            if f < f_opt:
                argmin = ix
                f_opt = f

        given[key] = val

        if pbar is not None:
            pbar.update(1)

    return f_opt, [all_keys[argmin]]


def _held_out_inner(
    engine: "Engine",
    fn: HoldOutFunc,
    search: HoldOutSearchMethod,
    n: int,
    values,
    given: dict[Union[str, int], Any],
    pbar: Optional[tqdm],
):
    if search == HoldOutSearchMethod.Greedy:
        return _held_out_inner_greedy(engine, fn, values, given, pbar)
    elif search == HoldOutSearchMethod.Enumerate:
        return _held_out_inner_enum(engine, fn, n, values, given, pbar)
    else:
        raise NotImplementedError


def _held_out_base(
    engine: "Engine",
    fn: HoldOutFunc,
    search: HoldOutSearchMethod,
    values,
    given: dict[Union[str, int], Any],
    quiet: bool = False,
) -> pl.DataFrame:
    if quiet:
        pbar = None
    else:
        n = len(given)
        if search == HoldOutSearchMethod.Greedy:
            total = n**2 / 2 - n
        elif search == HoldOutSearchMethod.Enumerate:
            total = 2**n
        pbar = tqdm(total=total)

    f = _held_out_compute(engine, fn, values, given)
    n = len(given)

    fs = [f]
    rm_keys = [None]
    keys_removed = [0]

    if not quiet:
        pbar.update(1)

    for i in range(n):
        f_opt, keys = _held_out_inner(
            engine, fn, search, i + 1, values, given, pbar
        )

        keys_removed.append(i + 1)

        fs.append(f_opt)
        rm_keys.append(keys)

        if search == HoldOutSearchMethod.Greedy:
            given.pop(next(iter(keys)))

    if not quiet:
        pbar.close()

    return pl.DataFrame(
        [
            pl.Series("feature_rmed", rm_keys),
            pl.Series(str(fn), fs),
            pl.Series("keys_rmed", keys_removed),
        ]
    )


def held_out_neglogp(
    engine: "Engine",
    values,
    given: dict[Union[str, int], Any],
    quiet: bool = False,
    greedy: bool = True,
) -> pl.DataFrame:
    r"""
    Compute -logp for values while sequentially dropping given conditions.

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
    quiet: bool
        Prevent the display of a progress bar.
    greedy: bool
        Use a greedy algorithm which is faster but may be less optimal.

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
    >>> given = (
    ...     satellites.df.to_pandas()
    ...     .set_index("ID")
    ...     .loc["Intelsat 903", :]
    ...     .dropna()
    ...     .to_dict()
    ... )
    >>> period = given.pop("Period_minutes")
    >>> held_out_neglogp(
    ...     satellites,
    ...     pl.Series("Period_minutes", [period]),
    ...     given,
    ...     quiet=True,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    shape: (19, 3)
    ┌─────────────────────────┬─────────────────────┬───────────┐
    │ feature_rmed            ┆ HoldOutFunc.NegLogp ┆ keys_rmed │
    │ ---                     ┆ ---                 ┆ ---       │
    │ list[str]               ┆ f64                 ┆ i64       │
    ╞═════════════════════════╪═════════════════════╪═══════════╡
    │ null                    ┆ 7.808063            ┆ 0         │
    │ ["Apogee_km"]           ┆ 5.082683            ┆ 1         │
    │ ["Eccentricity"]        ┆ 2.931816            ┆ 2         │
    │ ["Launch_Vehicle"]      ┆ 2.931816            ┆ 3         │
    │ …                       ┆ …                   ┆ …         │
    │ ["Power_watts"]         ┆ 2.932103            ┆ 15        │
    │ ["Inclination_radians"] ┆ 2.933732            ┆ 16        │
    │ ["Users"]               ┆ 2.940667            ┆ 17        │
    │ ["Perigee_km"]          ┆ 3.956759            ┆ 18        │
    └─────────────────────────┴─────────────────────┴───────────┘

    If we don't want to use the greedy search, we can enumerate, but we need to
    be mindful that the number of conditions we must enumerate over is 2^n

    >>> keys = sorted(list(given.keys()))
    >>> _ = [given.pop(c) for c in keys[-10:]]
    >>> held_out_neglogp(
    ...     satellites,
    ...     pl.Series("Period_minutes", [period]),
    ...     given,
    ...     quiet=True,
    ...     greedy=False,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    shape: (9, 3)
    ┌───────────────────────────────────┬─────────────────────┬───────────┐
    │ feature_rmed                      ┆ HoldOutFunc.NegLogp ┆ keys_rmed │
    │ ---                               ┆ ---                 ┆ ---       │
    │ list[str]                         ┆ f64                 ┆ i64       │
    ╞═══════════════════════════════════╪═════════════════════╪═══════════╡
    │ null                              ┆ 7.853468            ┆ 0         │
    │ ["Apogee_km"]                     ┆ 5.106627            ┆ 1         │
    │ ["Apogee_km", "Eccentricity"]     ┆ 2.951662            ┆ 2         │
    │ ["Apogee_km", "Country_of_Operat… ┆ 2.951254            ┆ 3         │
    │ …                                 ┆ …                   ┆ …         │
    │ ["Apogee_km", "Country_of_Contra… ┆ 2.956224            ┆ 5         │
    │ ["Apogee_km", "Country_of_Contra… ┆ 2.96479             ┆ 6         │
    │ ["Apogee_km", "Country_of_Contra… ┆ 2.992173            ┆ 7         │
    │ ["Apogee_km", "Class_of_Orbit", … ┆ 3.956759            ┆ 8         │
    └───────────────────────────────────┴─────────────────────┴───────────┘
    """
    search = (
        HoldOutSearchMethod.Greedy if greedy else HoldOutSearchMethod.Enumerate
    )

    res = _held_out_base(
        engine,
        HoldOutFunc.NegLogp,
        search,
        values,
        deepcopy(given),
        quiet=quiet,
    )
    return res


def held_out_inconsistency(
    engine: "Engine",
    values,
    given: dict[Union[str, int], Any],
    quiet: bool = False,
    greedy: bool = True,
) -> pl.DataFrame:
    r"""
    Compute inconsistency for values while sequentially dropping given conditions.

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
    quiet: bool
        Prevent the display of a progress bar.
    greedy: bool
        Use a greedy algorithm which is faster but may be less optimal.

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
    >>> given = (
    ...     satellites.df.to_pandas()
    ...     .set_index("ID")
    ...     .loc["Intelsat 903", :]
    ...     .dropna()
    ...     .to_dict()
    ... )
    >>> period = given.pop("Period_minutes")
    >>> held_out_inconsistency(
    ...     satellites,
    ...     pl.Series("Period_minutes", [period]),
    ...     given,
    ...     quiet=True,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    shape: (19, 3)
    ┌─────────────────────────┬───────────────────────────┬───────────┐
    │ feature_rmed            ┆ HoldOutFunc.Inconsistency ┆ keys_rmed │
    │ ---                     ┆ ---                       ┆ ---       │
    │ list[str]               ┆ f64                       ┆ i64       │
    ╞═════════════════════════╪═══════════════════════════╪═══════════╡
    │ null                    ┆ 1.973348                  ┆ 0         │
    │ ["Apogee_km"]           ┆ 1.284557                  ┆ 1         │
    │ ["Eccentricity"]        ┆ 0.740964                  ┆ 2         │
    │ ["Launch_Vehicle"]      ┆ 0.740964                  ┆ 3         │
    │ …                       ┆ …                         ┆ …         │
    │ ["Power_watts"]         ┆ 0.741036                  ┆ 15        │
    │ ["Inclination_radians"] ┆ 0.741448                  ┆ 16        │
    │ ["Users"]               ┆ 0.743201                  ┆ 17        │
    │ ["Perigee_km"]          ┆ 1.0                       ┆ 18        │
    └─────────────────────────┴───────────────────────────┴───────────┘

    If we don't want to use the greedy search, we can enumerate, but we need to
    be mindful that the number of conditions we must enumerate over is 2^n

    >>> keys = sorted(list(given.keys()))
    >>> _ = [given.pop(c) for c in keys[-10:]]
    >>> held_out_inconsistency(
    ...     satellites,
    ...     pl.Series("Period_minutes", [period]),
    ...     given,
    ...     quiet=True,
    ...     greedy=False,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    shape: (9, 3)
    ┌───────────────────────────────────┬───────────────────────────┬───────────┐
    │ feature_rmed                      ┆ HoldOutFunc.Inconsistency ┆ keys_rmed │
    │ ---                               ┆ ---                       ┆ ---       │
    │ list[str]                         ┆ f64                       ┆ i64       │
    ╞═══════════════════════════════════╪═══════════════════════════╪═══════════╡
    │ null                              ┆ 1.984823                  ┆ 0         │
    │ ["Apogee_km"]                     ┆ 1.290609                  ┆ 1         │
    │ ["Apogee_km", "Eccentricity"]     ┆ 0.74598                   ┆ 2         │
    │ ["Apogee_km", "Country_of_Operat… ┆ 0.745877                  ┆ 3         │
    │ …                                 ┆ …                         ┆ …         │
    │ ["Apogee_km", "Country_of_Contra… ┆ 0.747133                  ┆ 5         │
    │ ["Apogee_km", "Country_of_Contra… ┆ 0.749297                  ┆ 6         │
    │ ["Apogee_km", "Country_of_Contra… ┆ 0.756218                  ┆ 7         │
    │ ["Apogee_km", "Class_of_Orbit", … ┆ 1.0                       ┆ 8         │
    └───────────────────────────────────┴───────────────────────────┴───────────┘
    """
    search = (
        HoldOutSearchMethod.Greedy if greedy else HoldOutSearchMethod.Enumerate
    )

    res = _held_out_base(
        engine,
        HoldOutFunc.Inconsistency,
        search,
        values,
        deepcopy(given),
        quiet=quiet,
    )
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()
