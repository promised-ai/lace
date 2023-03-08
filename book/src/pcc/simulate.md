# Simulating data

If you've used `logp`, you already understand how to `simulate` data. In both
`logp` and `simulate` you define a distribution. In `logp` the output is an
evaluation of a specific point (or points) in the distribution; in `simulate`
you generate from the distribution.

We can simulate from joint distributions

```python
from lace.examples import Animals

animals = Animals()

swims = animals.simulate(['swims'], n=10)
```

Or we can simulate from conditional distributions

```python
swims = animals.simulate(['swims'], given={'flippers': 1}, n=10)
```

If we want to create a debiased dataset we can do something like this: There
are too many land animals in the animals dataset, we'd like an even
representation of land and aquatic animals. All we need to do is simulate from
the conditionals and concatenate the results.

```python
import polars as pl

n = animals.n_rows

target_col = 'swims'
other_cols = [col for col in animals.columns if col != target_col]

land_animals = animals.simulate(
    other_cols,
    given={target_col: 0},
    n=n//2,
    include_given=True
)

aquatic_animals = animals.simulate(
    other_cols,
    given={target_col: 1},
    n=n//2,
    include_given=True
)

df = pl.concat([land_animals, aquatic_animals])
```

That's it! We introduced a new keyword argument, `include_given`, which
includes the `given` conditions in the output so we don't have to add them back
manually.

## The `draw` method

The `draw` method is the in-table version of `simulate`. `draw` takes the row
and column indices and produces values from the probability distribution
describing that specific cell in the table.

```python
otter_swims = animals.draw('otter', 'swims', n=10)
```

## Evaluating simulated data

There are a number of ways to evaluate the quality of simulated (synthetic)
data:

-  Overlay histograms of synthetic data over histograms of the real data for
    each variable.
- Compare the correlation matrices emitted by the real and synthetic data.
- Train a classifier to classify real and synthetic data.  The better the
    synthetic data, the more difficult it will be for a classifier to identify
    synthetic data. Note that you must consider the precision of the data. Lace
    simulates full precision data. If the real data are rounded to a smaller
    number of decimal places, a classifier may pick up on that. To fix this,
    simply round the simulated data.
- Train a model on synthetic data and compare its performance on a real-data
    test set against a model trained on real data. Close performance to the
    real-data-trained model indicates higher quality synthetic data.

If you are concerned about sensitive information leakage, you should also
measure the similarity each synthetic record to each real record. Secure
synthetic data should not contain records that are so so to the originals that
they may reproduce sensitive information.
