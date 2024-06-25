"""Check the basics of the example datasets."""

from numpy.testing import assert_almost_equal

from lace import examples


def test_animals():
    """Check the output of the animals example."""

    engine = examples.Animals()

    assert engine.shape == (50, 85)
    swim_cat, swim_unc = engine.predict("swims")
    assert swim_cat == 0
    assert_almost_equal(swim_unc, 0.03782005724890601, 6)
