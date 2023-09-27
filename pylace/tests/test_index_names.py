import pytest

from lace.examples import Animals


@pytest.fixture(scope="module")
def animals():
    animals = Animals()
    animals.df = animals.df.to_pandas().set_index("id")
    return animals


def test_impute_index_name(animals):
    """Check impute returns the correct index name."""
    engine = animals.engine

    predicted = engine.impute(0, rows=engine.index)
    assert "index" in predicted.columns


def test_getitem_index_name(animals):
    """Check __getitem__ returns the correct index name."""
    engine = animals.engine

    df = engine["fierce"]
    assert "index" in df.columns
