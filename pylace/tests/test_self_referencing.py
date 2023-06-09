"""Tests whether engine functions work with various engine outputs."""
import random

import polars as pl
import pytest

from lace.examples import Animals, Satellites


@pytest.fixture(scope="module")
def satellites():
    return Satellites()


def test_multi_column_logp_on_simulate_output(satellites):
    xs = satellites.simulate(["Period_minutes", "Class_of_Orbit"], n=100)
    assert xs.shape == (100, 2)
    logps = satellites.logp(xs)
    assert logps.shape == (100,)


def test_single_column_logp_on_simulate_output(satellites):
    xs = satellites.simulate(["Class_of_Orbit"], n=100)
    assert xs.shape == (100, 1)
    logps = satellites.logp(xs)
    assert logps.shape == (100,)


def test_multi_column_logp_on_simulate_output_single(satellites):
    xs = satellites.simulate(["Period_minutes", "Class_of_Orbit"], n=1)
    assert xs.shape == (1, 2)
    logp = satellites.logp(xs)
    assert isinstance(logp, float)


def test_single_column_logp_on_simulate_output_single(satellites):
    xs = satellites.simulate(["Class_of_Orbit"], n=1)
    assert xs.shape == (1, 1)
    logp = satellites.logp(xs)
    assert isinstance(logp, float)


def test_logp_on_draw_output(satellites):
    for _ in range(100):
        col = random.choice(satellites.columns)
        row = random.choice(satellites.index)
        xs = satellites.draw(row=row, col=col, n=50)
        assert xs.shape == (50,)  # Series
        logps = satellites.logp(xs)
        assert logps.shape == (50,)


def test_surprisal_on_draw_output(satellites):
    for _ in range(100):
        col = random.choice(satellites.columns)
        row = random.choice(satellites.index)
        xs = satellites.draw(row=row, col=col, n=50)
        assert xs.shape == (50,)  # Series
        surp = satellites.surprisal(col=col, rows=[row], values=xs)
        assert surp.shape == (50,)


def test_surprisal_on_simulate_output(satellites):
    for _ in range(100):
        col = random.choice(satellites.columns)
        row = random.choice(satellites.index)
        xs = satellites.simulate([col], n=50)
        assert xs.shape == (50, 1)  # Series
        surp = satellites.surprisal(col=col, rows=[row], values=xs[col])
        assert surp.shape == (50,)


def test_append_cols_on_simulate_output(satellites):
    # append a satellites column to animals
    engine = Animals()
    cols = satellites.simulate(["Class_of_Orbit"], n=engine.shape[0])
    cols = cols.with_columns(pl.Series("index", engine.index))
    engine.append_columns(cols)
    assert engine.shape[1] == 86
