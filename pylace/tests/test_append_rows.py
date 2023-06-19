import pandas as pd
import polars as pl
import pytest

from lace.examples import Animals


@pytest.fixture()
def animals():
    animals = Animals()
    return animals


@pytest.mark.parametrize(
    "index_name", ["ID", "Id", "id", "Index", "index", "INDEX"]
)
def test_polars_with_index(animals, index_name):
    rows = pl.DataFrame(
        {
            "swims": [0, 1],
            "flys": [0, 1],
            index_name: ["unicorn", "flying+fish"],
        }
    )
    animals.append_rows(rows)
    assert animals["unicorn", "swims"] == 0
    assert animals["unicorn", "flys"] == 0
    assert animals["unicorn", "brown"] is None

    assert animals["flying+fish", "swims"] == 1
    assert animals["flying+fish", "flys"] == 1
    assert animals["flying+fish", "brown"] is None


def test_pandas_with_index(animals):
    rows = pd.DataFrame(
        {
            "swims": [0, 1],
            "flys": [0, 1],
        },
        index=["unicorn", "flying+fish"],
    )

    animals.append_rows(rows)
    assert animals["unicorn", "swims"] == 0
    assert animals["unicorn", "flys"] == 0
    assert animals["unicorn", "brown"] is None

    assert animals["flying+fish", "swims"] == 1
    assert animals["flying+fish", "flys"] == 1
    assert animals["flying+fish", "brown"] is None


def test_dict_dense(animals):
    rows = {
        "flying+fish": {"swims": 1, "flys": 1},
        "unicorn": {"swims": 0, "flys": 0},
    }

    animals.append_rows(rows)

    assert animals["unicorn", "swims"] == 0
    assert animals["unicorn", "flys"] == 0
    assert animals["unicorn", "brown"] is None

    assert animals["flying+fish", "swims"] == 1
    assert animals["flying+fish", "flys"] == 1
    assert animals["flying+fish", "brown"] is None


def test_error_on_polars_with_no_index(animals):
    rows = pl.DataFrame(
        {
            "swims": [0, 1],
            "flys": [0, 1],
            "Index": [51, 52],  # this is not a string
        }
    )
    with pytest.raises(ValueError, match="'Index' are not strings"):
        animals.append_rows(rows)
