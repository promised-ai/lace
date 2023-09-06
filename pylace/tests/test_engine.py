from copy import deepcopy

from lace import examples


def test_deep_copy():
    a = examples.Animals()
    b = deepcopy(a)

    assert a is not b

    assert a.columns == b.columns
    assert a.shape == b.shape
