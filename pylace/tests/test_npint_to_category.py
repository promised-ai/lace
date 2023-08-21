import numpy as np
import pytest

from lace.examples import Animals


@pytest.fixture(scope="module")
def animals():
    animals = Animals()
    animals.df = animals.df.to_pandas().set_index("id")
    return animals


def test_category_from_np_int(animals):
    """Check that np.int* can be conerted to a category."""

    for ty in [np.int64, np.int32, np.int16, np.int8]:
        animals.engine.simulate(
            ["solitary"],
            given={
                "smart": ty(0),
            },
        )
