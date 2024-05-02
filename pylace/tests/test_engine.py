from copy import deepcopy

from lace import examples


def test_deep_copy():
    a = examples.Animals()
    b = deepcopy(a)

    assert a is not b

    assert a.columns == b.columns
    assert a.shape == b.shape


def test_remove_rows():
    from lace.examples import Animals

    engine = Animals()
    n_rows = engine.n_rows
    removed = engine.remove_rows(["cow", "wolf"])
    assert n_rows == engine.n_rows + 2
    assert removed["index"].len() == 2
