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
    assert len(engine.index) == n_rows - 2
    assert removed["index"].len() == 2

    assert "wolf" not in engine.index
    assert "cow" not in engine.index


def test_diagnostics():
    from lace.examples import Animals

    engine = Animals()
    engine.diagnostics()


def test_readd_rows():
    from lace.examples import Animals

    engine = Animals()
    n_rows = engine.n_rows

    removed = engine.remove_rows(["cow", "wolf"])
    assert len(engine.index) == n_rows - 2
    assert n_rows == engine.n_rows + 2
    assert removed["index"].len() == 2
    assert "wolf" not in engine.index
    assert "cow" not in engine.index

    print(removed)
    engine.append_rows(removed)
    assert n_rows == engine.n_rows
    assert engine.index[-2] == "cow"
    assert engine.index[-1] == "wolf"
    assert len(engine.index) == n_rows
