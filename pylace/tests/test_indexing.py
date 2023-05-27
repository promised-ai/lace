import random
import pytest

from lace.examples import Animals


@pytest.fixture(scope="module")
def animals():
    return Animals()

def _random_index(n, strs):
    u = random.random()
    if u < 1/3:
        return random.randint(0, n-1)
    elif u < 2/3:
        return -random.randint(0, n-1)
    else:
        return random.choice(strs)


def tests_index_positive_int_tuple(animals):
    assert animals[0, 0] == 0
    assert animals[2, 3] == 0
    assert animals[49, 84] == 1

    for row in range(animals.n_rows):
        for col in range(animals.n_cols):
            assert animals[row, col] == animals.df.iloc[row, col]


def tests_index_negative_int_tuple(animals):
    assert animals[-1, -1] == 1
    assert animals[-49, 1] == 0
    assert animals[10, -11] == 1
    assert animals[-3, -11] == 1


def test_string_tuple_index(animals):
    assert animals["otter", "swims"] == 1
    assert animals["bat", "swims"] == 0
    assert animals["bat", "flys"] == 1

    for row in animals.index:
        for col in animals.columns:
            assert animals[row, col] == animals.df.loc[row, col]


def test_tuple_index_fuzzy_smoke(animals):
    row_strs = animals.index
    col_strs = animals.columns
    n_rows = len(row_strs)
    n_cols = len(col_strs)

    for _ in range(1000):
        row = _random_index(n_rows, row_strs)
        col = _random_index(n_cols, col_strs)
        if isinstance(col, str) and isinstance(row, int):
            assert animals[row, col] == animals.df[col].iloc[row]
        elif isinstance(col, str) and isinstance(row, str):
            assert animals[row, col] == animals.df.loc[row, col]
        elif isinstance(col, int) and isinstance(row, int):
            assert animals[row, col] == animals.df.iloc[row, col]
        else:
            assert animals[row, col] == animals.df.iloc[col].loc[row]
