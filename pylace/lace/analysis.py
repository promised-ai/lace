"""Tools for analysis of probabilistic cross-categorization results in Lace."""


import enum
import itertools as it
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import polars as pl
from tqdm import tqdm

from lace import utils

ABLATIVE_ERR = "ablative-err"
ABLATIVE_DIST = "ablative-dist"
PRED_EXPLAIN_METHODS = [ABLATIVE_ERR, ABLATIVE_DIST]


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
    Uncertainty = 2

    def __repr__(self) -> str:
        """Return the canonical string representation of the object."""
        if self == HoldOutFunc.NegLogp:
            return "-logp"
        elif self == HoldOutFunc.Inconsistency:
            return "inconsistency"
        elif self == HoldOutFunc.Uncertainty:
            return "uncertainty"
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
    elif fn == HoldOutFunc.Uncertainty:
        return engine.predict(values, given=given)[1]
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


def held_out_uncertainty(
    engine: "Engine",
    target: Union[str, int],
    given: dict[Union[str, int], Any],
    quiet: bool = False,
    greedy: bool = True,
) -> pl.DataFrame:
    r"""
    Compute prediction uncertainty while sequentially dropping given conditions.

    Parameters
    ----------
    engine: Engine
        The Engine used to compute inconsistency
    target: str or int
        The target column for prediction
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
        A DataFrame with a 'feature' column and a uncertainty column.

    Examples
    --------
    >>> import polars as pl
    >>> from lace.examples import Satellites
    >>> from lace.analysis import held_out_uncertainty
    >>> satellites = Satellites()
    >>> given = (
    ...     satellites.df.to_pandas()
    ...     .set_index("ID")
    ...     .loc["Intelsat 903", :]
    ...     .dropna()
    ...     .to_dict()
    ... )
    >>> period = given.pop("Period_minutes")
    >>> held_out_uncertainty(
    ...     satellites,
    ...     "Period_minutes",
    ...     given,
    ...     quiet=True,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    shape: (19, 3)
    ┌──────────────────────────────────┬─────────────────────────┬───────────┐
    │ feature_rmed                     ┆ HoldOutFunc.Uncertainty ┆ keys_rmed │
    │ ---                              ┆ ---                     ┆ ---       │
    │ list[str]                        ┆ f64                     ┆ i64       │
    ╞══════════════════════════════════╪═════════════════════════╪═══════════╡
    │ null                             ┆ 0.543803                ┆ 0         │
    │ ["Perigee_km"]                   ┆ 0.543803                ┆ 1         │
    │ ["Source_Used_for_Orbital_Data"] ┆ 0.543799                ┆ 2         │
    │ ["Class_of_Orbit"]               ┆ 0.5428                  ┆ 3         │
    │ …                                ┆ …                       ┆ …         │
    │ ["Purpose"]                      ┆ 0.440191                ┆ 15        │
    │ ["Eccentricity"]                 ┆ 0.438648                ┆ 16        │
    │ ["Inclination_radians"]          ┆ 0.44004                 ┆ 17        │
    │ ["Apogee_km"]                    ┆ 1.443993                ┆ 18        │
    └──────────────────────────────────┴─────────────────────────┴───────────┘

    If we don't want to use the greedy search, we can enumerate, but we need to
    be mindful that the number of conditions we must enumerate over is 2^n

    >>> keys = sorted(list(given.keys()))
    >>> _ = [given.pop(c) for c in keys[-10:]]
    >>> held_out_uncertainty(
    ...     satellites,
    ...     "Period_minutes",
    ...     given,
    ...     quiet=True,
    ...     greedy=False,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    shape: (9, 3)
    ┌───────────────────────────────────┬─────────────────────────┬───────────┐
    │ feature_rmed                      ┆ HoldOutFunc.Uncertainty ┆ keys_rmed │
    │ ---                               ┆ ---                     ┆ ---       │
    │ list[str]                         ┆ f64                     ┆ i64       │
    ╞═══════════════════════════════════╪═════════════════════════╪═══════════╡
    │ null                              ┆ 0.579697                ┆ 0         │
    │ ["Expected_Lifetime"]             ┆ 0.553909                ┆ 1         │
    │ ["Class_of_Orbit", "Expected_Lif… ┆ 0.395509                ┆ 2         │
    │ ["Class_of_Orbit", "Date_of_Laun… ┆ 0.38756                 ┆ 3         │
    │ …                                 ┆ …                       ┆ …         │
    │ ["Class_of_Orbit", "Country_of_O… ┆ 0.399052                ┆ 5         │
    │ ["Class_of_Orbit", "Country_of_O… ┆ 0.41235                 ┆ 6         │
    │ ["Class_of_Orbit", "Country_of_C… ┆ 0.44004                 ┆ 7         │
    │ ["Apogee_km", "Class_of_Orbit", … ┆ 1.443993                ┆ 8         │
    └───────────────────────────────────┴─────────────────────────┴───────────┘
    """
    search = (
        HoldOutSearchMethod.Greedy if greedy else HoldOutSearchMethod.Enumerate
    )

    res = _held_out_base(
        engine,
        HoldOutFunc.Uncertainty,
        search,
        target,
        deepcopy(given),
        quiet=quiet,
    )
    return res


def _attributable_holdout(
    engine: "Engine",
    fn: HoldOutFunc,
    values,
    given: dict[Union[str, int], Any],
    quiet: bool = False,
    greedy: bool = True,
):
    search = (
        HoldOutSearchMethod.Greedy if greedy else HoldOutSearchMethod.Enumerate
    )

    fn_str = str(fn)

    res = _held_out_base(
        engine, fn, search, values, deepcopy(given), quiet=quiet
    )

    n_holdouts = res.shape[0]

    if n_holdouts < 2:
        return 0.0, res

    area = 0.0
    fn_max = res[fn_str][0]

    fn_a = res[fn_str][1]
    area += (fn_a / fn_max) / 2.0

    n = 2

    for i in range(2, n_holdouts):
        fn_b = res[fn_str][i]

        if fn_b > fn_max:
            break

        area += ((fn_max - fn_b) / fn_max) / 2.0
        fn_a = fn_b
        n += 1

    if n <= 2:
        return 0, res

    return area / n, res


def attributable_inconsistency(
    engine: "Engine",
    values,
    given: dict[Union[str, int], Any],
    quiet: bool = False,
    greedy: bool = True,
) -> Tuple[float, pl.DataFrame]:
    r"""
    Determine what fraction of inconsistency is attributable.

    The fraction will be higher if dropping fewer predictor reduces
    inconsistency quickly. The fraction will be 1 if one predictor drops
    inconsistency to zero (this is unlikely to ever occur). The fraction will
    be 0 if dropping predictors has no effect.

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
    float
        The fraction [0, 1] of the inconsistency that is attributable
    polars.DataFrame
        The result of held_out_inconsistency

    Examples
    --------
    >>> import polars as pl
    >>> from lace.examples import Satellites
    >>> from lace.analysis import attributable_inconsistency
    >>> satellites = Satellites()
    >>> given = (
    ...     satellites.df.to_pandas()
    ...     .set_index("ID")
    ...     .loc["Intelsat 903", :]
    ...     .dropna()
    ...     .to_dict()
    ... )
    >>> period = given.pop("Period_minutes")
    >>> frac, df = attributable_inconsistency(
    ...     satellites,
    ...     pl.Series("Period_minutes", [period]),
    ...     given,
    ...     quiet=True,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    >>> frac
    0.2930260843667006
    """

    return _attributable_holdout(
        engine,
        HoldOutFunc.Inconsistency,
        values,
        given,
        quiet=quiet,
        greedy=greedy,
    )


def attributable_neglogp(
    engine: "Engine",
    values,
    given: dict[Union[str, int], Any],
    quiet: bool = False,
    greedy: bool = True,
) -> Tuple[float, pl.DataFrame]:
    r"""
    Determine what fraction of surprisal (-log p) is attributable.

    The fraction will be higher if dropping fewer predictor reduces surprisal
    quickly. The fraction will be 1 if one predictor drops surprisal to zero
    (this can never occur). The fraction will be 0 if dropping predictors has
    no effect.

    Parameters
    ----------
    engine: Engine
        The Engine used to compute inconsistency
    values: polars or pandas DataFrame or Series
        The values over which to compute the -log p. Each row of the
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
    float
        The fraction [0, 1] of the surprisal that is attributable
    polars.DataFrame
        The result of held_out_neglogp

    Examples
    --------
    >>> import polars as pl
    >>> from lace.examples import Satellites
    >>> from lace.analysis import attributable_neglogp
    >>> satellites = Satellites()
    >>> given = (
    ...     satellites.df.to_pandas()
    ...     .set_index("ID")
    ...     .loc["Intelsat 903", :]
    ...     .dropna()
    ...     .to_dict()
    ... )
    >>> period = given.pop("Period_minutes")
    >>> frac, df =  attributable_neglogp(
    ...     satellites,
    ...     pl.Series("Period_minutes", [period]),
    ...     given,
    ...     quiet=True,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    >>> frac
    0.29302608436670047
    """

    return _attributable_holdout(
        engine,
        HoldOutFunc.NegLogp,
        values,
        given,
        quiet=quiet,
        greedy=greedy,
    )


def attributable_uncertainty(
    engine: "Engine",
    target: Union[str, int],
    given: dict[Union[str, int], Any],
    quiet: bool = False,
    greedy: bool = True,
) -> Tuple[float, pl.DataFrame]:
    r"""
    Determine what fraction of uncertainty is attributable.

    The fraction will be higher if dropping fewer predictor reduces uncertainty
    quickly. The fraction will be 1 if one predictor drops uncertainty to zero
    (this is unlikely). The fraction will be 0 if dropping predictors has no
    effect.

    Parameters
    ----------
    engine: Engine
        The Engine used to compute inconsistency
    target: str or int
        The prediction target
    given: dict[index, value], optional
        A dictionary mapping column indices/name to values, which specifies
        conditions on the observations.
    quiet: bool
        Prevent the display of a progress bar.
    greedy: bool
        Use a greedy algorithm which is faster but may be less optimal.

    Returns
    -------
    float
        The fraction [0, 1] of the uncertainty that is attributable
    polars.DataFrame
        The result of held_out_uncertainty

    Examples
    --------
    >>> import polars as pl
    >>> from lace.examples import Satellites
    >>> from lace.analysis import attributable_uncertainty
    >>> satellites = Satellites()
    >>> given = (
    ...     satellites.df.to_pandas()
    ...     .set_index("ID")
    ...     .loc["Intelsat 903", :]
    ...     .dropna()
    ...     .to_dict()
    ... )
    >>> period = given.pop("Period_minutes")
    >>> frac, df =  attributable_uncertainty(
    ...     satellites,
    ...     "Period_minutes",
    ...     given,
    ...     quiet=True,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    >>> frac
    0.09788266843531362
    """

    return _attributable_holdout(
        engine,
        HoldOutFunc.Uncertainty,
        target,
        given,
        quiet=quiet,
        greedy=greedy,
    )


def _explain_ablative_err(
    engine: "Engine",
    target: Union[int, str],
    given: dict[Union[str, int], Any],
):
    xs = utils.predict_xs(engine, target, None, mass=0.995)

    ftype = engine.ftype(target)
    if ftype == "Categorial":
        norm = 1.0 / len(xs)
    elif ftype in ("Continuous", "Count"):
        norm = xs[1] - xs[0]
    else:
        raise ValueError(f"Unupported FType `{ftype}`")

    baseline = engine.logp(xs, given).exp()

    cols = []
    imps = []
    for k in list(given.keys()):
        # remove the value
        val = given.pop(k)
        ps = engine.logp(xs, given).exp()

        err = (ps - baseline).sum() * norm

        # put the value back
        given[k] = val

        cols.append(k)
        imps.append(err)

    return cols, imps


def _explain_ablative_dist(
    engine: "Engine",
    target: Union[int, str],
    given: dict[Union[str, int], Any],
):
    ftype = engine.ftype(target)
    if ftype != "Continuous":
        msg = (
            "`ablative-dist` explanation only valid for Continuous targets"
            f" but target `{target}` is `{ftype}`"
        )
        raise ValueError(msg)

    cols = []
    imps = []
    for k in list(given.keys()):
        baseline = engine.predict(target, given, with_uncertainty=False)

        # remove the value
        val = given.pop(k)

        pred = engine.predict(target, given, with_uncertainty=False)

        # put the value back
        given[k] = val

        cols.append(k)
        imps.append(pred - baseline)

    return cols, imps


def explain_prediction(
    engine: "Engine",
    target: Union[int, str],
    given: dict[Union[str, int], Any],
    *,
    method: Optional[str] = None,
):
    """
    Explain the relevance of each predictor when predicting a target.

    Parameters
    ----------
    engine: lace.Engine
        The source engine
    target: str, int
        The target variable -- the variable to predict
    given: Dict[index, value], optional
        A dictionary mapping column indices/name to values, which specifies
        conditions on the observations.
    method: str, optional
        The method to use for explanation:
        * 'ablative-err' (default): computes the different between p(y|X) and
          p(x|X - xᵢ) for each predictor xᵢ in the `given`, X.
        * 'ablative-dist': computed the error between the predictions (argmax)
          of p(y|X) and p(x|X - xᵢ) for each predictor xᵢ in the `given`, X. Note
          that this method does not support categorical targets.

    Returns
    -------
    cols: List[str]
        The column names associated with each importance
    imps: List[float]
        The list of importances for each column

    Examples
    --------
    >>> import polars as pl
    >>> from lace.examples import Satellites
    >>> from lace.analysis import explain_prediction
    >>> engine = Satellites()

    Define a target

    >>> target = 'Period_minutes'

    We'll use a row from the data

    >>> row = engine[5, :].to_dicts()[0]
    >>> ix = row.pop('index')
    >>> _ = row.pop(target)
    >>> given = { k: v for k, v in row.items() if v is not None }

    The default importance method, 'ablative-err', measures the error between
    the baseline predictive distribution, and the distribution when a predictor
    is dropped.

    >>> cols, imps = explain_prediction(
    ...     engine,
    ...     target,
    ...     given,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    >>> pl.DataFrame({'col': cols, 'imp': imps})
    shape: (18, 2)
    ┌──────────────────────────────┬─────────────┐
    │ col                          ┆ imp         │
    │ ---                          ┆ ---         │
    │ str                          ┆ f64         │
    ╞══════════════════════════════╪═════════════╡
    │ Country_of_Operator          ┆ 3.5216e-16  │
    │ Users                        ┆ -3.1668e-14 │
    │ Purpose                      ┆ -9.5636e-14 │
    │ Class_of_Orbit               ┆ -1.8263e-15 │
    │ …                            ┆ …           │
    │ Launch_Site                  ┆ -2.8416e-15 │
    │ Launch_Vehicle               ┆ 1.0704e-14  │
    │ Source_Used_for_Orbital_Data ┆ -3.9301e-15 │
    │ Inclination_radians          ┆ -9.6259e-15 │
    └──────────────────────────────┴─────────────┘

    Get the importances using the 'ablative-dist' method, which measures how
    much the prediction would change if a predictor was dropped.

    >>> cols, imps = explain_prediction(
    ...     engine,
    ...     target,
    ...     given,
    ...     method='ablative-dist'
    ... )  # doctest: +NORMALIZE_WHITESPACE
    >>> pl.DataFrame({'col': cols, 'imp': imps})
    shape: (18, 2)
    ┌──────────────────────────────┬───────────┐
    │ col                          ┆ imp       │
    │ ---                          ┆ ---       │
    │ str                          ┆ f64       │
    ╞══════════════════════════════╪═══════════╡
    │ Country_of_Operator          ┆ -0.000109 │
    │ Users                        ┆ 0.081289  │
    │ Purpose                      ┆ 0.18938   │
    │ Class_of_Orbit               ┆ 0.000133  │
    │ …                            ┆ …         │
    │ Launch_Site                  ┆ 0.003411  │
    │ Launch_Vehicle               ┆ -0.018817 │
    │ Source_Used_for_Orbital_Data ┆ 0.001454  │
    │ Inclination_radians          ┆ 0.057333  │
    └──────────────────────────────┴───────────┘
    """
    if method is None:
        method = ABLATIVE_ERR

    if method == ABLATIVE_ERR:
        return _explain_ablative_err(engine, target, given)
    elif method == ABLATIVE_DIST:
        return _explain_ablative_dist(engine, target, given)
    else:
        raise ValueError(
            f"Invalid method `{method}`, valid methods are {PRED_EXPLAIN_METHODS}"
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
