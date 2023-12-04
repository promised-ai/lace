import random

import pytest

from lace.examples import Animals


@pytest.fixture(scope="module")
def animals():
    animals = Animals()
    animals.df = animals.df.to_pandas().set_index("id")
    return animals


def _random_index(n, strs):
    u = random.random()
    if u < 1 / 3:
        return random.randint(0, n - 1)
    elif u < 2 / 3:
        return -random.randint(0, n - 1)
    else:
        return random.choice(strs)


@pytest.mark.parametrize("target", ["black", "swims", 12, 45])
def test_single_index_consistency(target):
    # for some reason, the order of the row indeices iterator was different
    # each time we read in the metadata -- probably because of some random
    # state initialization in a hashmap. Not sure why that would happen, but
    # we fixed that particular issue.
    a1 = Animals()
    a2 = Animals()
    xs = a1[target][:, 1]
    ys = a2[target][:, 1]
    assert all(x == y for x, y in zip(xs, ys))


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


def test_poscont_col_slice_indexing_1(animals):
    data = animals["otter", :2]
    columns = data.columns

    assert data.shape == (1, 3)
    assert columns[1] == animals.columns[0]
    assert columns[2] == animals.columns[1]


def test_poscont_col_slice_indexing_2(animals):
    data = animals["otter", 80:]
    columns = data.columns

    assert data.shape == (1, 6)
    assert columns[1] == animals.columns[80]
    assert columns[2] == animals.columns[81]
    assert columns[3] == animals.columns[82]
    assert columns[4] == animals.columns[83]
    assert columns[5] == animals.columns[84]


def test_negcont_col_slice_indexing_1(animals):
    data = animals["otter", -2:]
    columns = data.columns

    assert data.shape == (1, 3)
    assert columns[1] == animals.columns[83]
    assert columns[2] == animals.columns[84]


def test_negcont_col_slice_indexing_2(animals):
    data = animals["otter", :-80]
    columns = data.columns

    assert data.shape == (1, 6)
    assert columns[1] == animals.columns[0]
    assert columns[2] == animals.columns[1]
    assert columns[3] == animals.columns[2]
    assert columns[4] == animals.columns[3]
    assert columns[5] == animals.columns[4]


def test_skip_col_slice_indexing_1(animals):
    data = animals["otter", 0:4:2]
    columns = data.columns

    assert data.shape == (1, 3)
    assert columns[1] == animals.columns[0]
    assert columns[2] == animals.columns[2]


def test_skip_col_slice_indexing_2(animals):
    data = animals["otter", 4:0:-1]
    columns = data.columns

    assert data.shape == (1, 5)
    assert columns[1] == animals.columns[4]
    assert columns[2] == animals.columns[3]
    assert columns[3] == animals.columns[2]
    assert columns[4] == animals.columns[1]


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
            assert animals[row, col] == animals.df.iloc[:, col].loc[row]
