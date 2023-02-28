# Conduct an analysis

You've made a codebook, you've fit a model, now you're ready to do learn.


Let's use the built-in examples to walk through some key concepts. The
`Animals` example isn't the biggest, or most complex, and that's exactly why
it's so great. People have acquired a ton of intuition about animals like how
and why you might categorize animals into a taxonomy, and why animals have
certain features and what that might tell us about other features of animals.
This means, that we can see if lace recovers our intuition.

<div class="multilang">

```python
from lace import examples

# if this is your first time using the example, lace must
# build the metadata
animals = examples.Animals()
```

```rust,noplayground
use lace::examples::Example

// You can create an Engine or an Oracle. An Oracle is
// basically an immutable Engine. You cannot add/edit data or
// extend runs (update).
let animals = Example::Animals.engine().unwrap();
```

</div>

Usually, the first question we want to ask of a new dataset is "What questions
can I answer?" This is a question about statistical dependence. Which features
of our dataset share statistical dependence with which others? This is closely
linked with the question "which things can I predict given which other things?"


In python, we can generate a plotly heatmap of *dependence probability*.

<p class="multilang">

```python
animals.clustermap(
    'depprob',
    color_continuous_scale='greys',
    zmin=0,
    zmax=1
).figure.show()
```

{{#include ../pcc/html/animals-depprob.html}}

In rust, we ask about dependence probabilities between individual pairs of
features

```rust
depprob_flippers = animals.depprob(
    "swims",
    "flippers",
).unwrap()
```

</p>
