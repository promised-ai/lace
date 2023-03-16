# Conduct an analysis

You've made a codebook, you've fit a model, now you're ready to do learn.


Let's use the built-in examples to walk through some key concepts. The
`Animals` example isn't the biggest, or most complex, and that's exactly why
it's so great. People have acquired a ton of intuition about animals like how
and why you might categorize animals into a taxonomy, and why animals have
certain features and what that might tell us about other features of animals.
This means, that we can see if lace recovers our intuition.

<div class=tabbed-blocks>

```python
from lace import examples

# if this is your first time using the example, lace must
# build the metadata
animals = examples.Animals()
```

```rust,noplayground
use lace::examples::Example
use lace::prelude::*;

// You can create an Engine or an Oracle. An Oracle is
// basically an immutable Engine. You cannot add/edit data or
// extend runs (update).
let animals = Example::Animals.engine().unwrap();
```
</div>

## Statistical structure

Usually, the first question we want to ask of a new dataset is "What questions
can I answer?" This is a question about statistical dependence. Which features
of our dataset share statistical dependence with which others? This is closely
linked with the question "which things can I predict given which other things?"


In python, we can generate a plotly heatmap of *dependence probability*.

<div class=tabbed-blocks>

```python
animals.clustermap(
    'depprob',
    color_continuous_scale='greys',
    zmin=0,
    zmax=1
).figure.show()
```
</div>

{{#include ../pcc/html/animals-depprob.html}}

In rust, we ask about dependence probabilities between individual pairs of
features

<div class=tabbed-blocks>

```rust,noplayground
let depprob_flippers = animals.depprob(
    "swims",
    "flippers",
).unwrap()
```
</div>

## Prediction

Now that we know which columns are predictive of each other, let's do some
predicting. We'll predict whether an animal swims. Just *an* animals. Not an
animals with flippers, or a tail. Any animal.

<div class=tabbed-blocks>

```python
animals.predict("swims")
```

```rust,noplayground
animals.predict(
    "swims",
    &Given::Nothing,
    Some(PredictUncertaintyType::JsDivergence),
    None,
)
```

</div>

Which outputs

```python
(0, 0.008287057807910558)
```

The first number is the prediction. Lace predicts that *an* animal does not
swims (because most of the animals in the dataset do not swim). The second
number is the *uncertainty*. Uncertainty is a number between 0 and 1
representing the disagreement between states. Uncertainty is 0 if all the
states completely agree on how to model the prediction, and is 1 if all the
states completely disagree. Note that uncertainty is not tied to variance.

The uncertainty of this prediction is very low.

We can add conditions. Let's predict whether an animal swims given that it has
flippers.

<div class=tabbed-blocks>

```python
animals.predict("swims", given={'flippers': 1})
```

```rust,noplayground
animals.predict(
    "swims",
    &Given::Conditions(vec![
        ("flippers": Datum::Categorical(1))
    ]),
    Some(PredictUncertaintyType::JsDivergence),
    None,
)
```
</div>

Output:

```python
(1, 0.05008037071634858)
```

The uncertainty is higher, but still quite low.

Let's add some more conditions that are indicative of a swimming animal and see
how that effects the uncertainty.

<div class=tabbed-blocks>

```python
animals.predict("swims", given={'flippers': 1, 'water': 1})
```

```rust,noplayground
animals.predict(
    "swims",
    &Given::Conditions(vec![
        ("flippers": Datum::Categorical(1)),
        ("water": Datum::Categorical(1)),
    ]),
    Some(PredictUncertaintyType::JsDivergence),
    None,
)
```
</div>

Output:

```python
(1, 0.05116664361415335)
```

The uncertainty is basically the same.

How about we try to mess with Lace? Let's try to confuse it by asking it to
predict whether an animal with flippers that does not go in the water swims.

<div class=tabbed-blocks>

```python
animals.predict("swims", given={'flippers': 1, 'water': 0})
```

```rust,noplayground
animals.predict(
    "swims",
    &Given::Conditions(vec![
        ("flippers": Datum::Categorical(1)),
        ("water": Datum::Categorical(0)),
    ]),
    Some(PredictUncertaintyType::JsDivergence),
    None,
)
```
</div>

Output:

```python
(0, 0.32863593091906085)
```

The uncertainty is really high! We've successfully confused lace.

## Evaluating likelihoods

Let's compute the likemportlihood to see what is going on

<div class=tabbed-blocks>

```python
import polars as pl

animals.logp(
    pl.Series("swims", [0, 1]),
    given={'flippers': 1, 'water': 0}
).exp()
```

```rust,noplayground
animals.logp(
    ["swims"],
    &[
        vec![Datum::Categorical(0)],
        vec![Datum::Categorical(1)],
    ],
    &Given::Conditions(vec![
        ("flippers": Datum::Categorical(1)),
        ("water": Datum::Categorical(0)),
    ]),
    None,
)
.unwrap()
.iter()
.map(|&logp| logp.exp())
.collect::<Vec<_>>()
```
</div>

Output:

```python
# polars
shape: (2,)
Series: 'logp' [f64]
[
	0.589939
	0.410061
]
```

## Anomaly detection

<div class=tabbed-blocks>

```python
animals.surprisal("fierce")\
    .sort("surprisal", descending=True)\
    .head(10)
```
</div>

Output:

```python
# poalrs
shape: (10, 3)
┌──────────────┬────────┬───────────┐
│ index        ┆ fierce ┆ surprisal │
│ ---          ┆ ---    ┆ ---       │
│ str          ┆ u32    ┆ f64       │
╞══════════════╪════════╪═══════════╡
│ pig          ┆ 1      ┆ 1.565845  │
│ rhinoceros   ┆ 1      ┆ 1.094639  │
│ buffalo      ┆ 1      ┆ 1.094639  │
│ chihuahua    ┆ 1      ┆ 0.802085  │
│ ...          ┆ ...    ┆ ...       │
│ collie       ┆ 0      ┆ 0.594919  │
│ otter        ┆ 0      ┆ 0.386639  │
│ hippopotamus ┆ 0      ┆ 0.328759  │
│ persian+cat  ┆ 0      ┆ 0.322771  │
└──────────────┴────────┴───────────┘
```
