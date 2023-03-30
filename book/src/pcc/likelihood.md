<style>
    #line-plot {
        aspect-ratio: 4/3;
    }
</style>

# Likelihood

Computing likelihoods is the bread and butter of Lace. Apart from the
clustering-based quantities, pretty much everything in Lace is computing
likelihoods behind the scenes. Prediction is finding the argmax of a
likelihood, surprisal is the negative likelihood, entropy is the integral of
the production of the likelihood and log likelihood.

Computing likelihood is simple. First, we'll pull in the Satellites example.

<div class=tabbed-blocks>

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lace import examples

satellites = examples.Satellites()
```
</div>

We'll compute the univariate likelihood of the `Period_minutes` feature over a
range of values. We'll compute \\(\log p(Period)\\) and the conditional log
likelihood of period given that the satellite is geosynchronous, \\( \log
p(Period | GEO)\\).

<div class=tabbed-blocks>

```python
period = pd.Series(np.linspace(0, 1500, 300), name="Period_minutes")

logp_period = satellies.logp(period)
logp_period_geo = satellies.logp(period, given={"Class_of_Orbit": "GEO"})
```
</div>

Rendered:

{{#include html/univariate-loglike.html}}

We can, of course compute likelihoods over multiple values:

<div class=tabbed-blocks>

```python
values = pd.DataFrame({
    'Period_minutes': [70.0, 320.0, 1440.0],
    'Class_of_Orbit': ['LEO', 'MEO', 'GEO'],
})

values['logp'] = satellites.logp(values).exp()
values
```
</div>

Output (values):

|    | Class_of_Orbit   |   Period_minutes |        logp |
|---:|:-----------------|-----------------:|------------:|
|  0 | LEO              |               70 | 0.000364503 |
|  1 | MEO              |              320 | 1.8201e-05  |
|  2 | GEO              |             1440 | 0.0158273   |


We can find a close proximity to multivariate prediction by combining
`simulate` and `logp`.

<div class=tabbed-blocks>

```python
# The things we observe
conditions = {
    'Class_of_Orbit': 'LEO',
    'Period_minutes': 80.0,
    'Launch_Vehicle': 'Proton',
}

# Simulate many, many values
simulated = satellites.simulate(
    ['Country_of_Operator', 'Purpose', 'Expected_Lifetime'],
    given=conditions,
    n=100000,  # big number
)

# compute the log likelihood of each draw given the conditions
logps = satellits.logp(simulated, given=conditions)

# return the draw with the highest likelihood
simualted[logps.arg_max()]
```
</div>

Output:

```
shape: (1, 3)
┌─────────────────────┬────────────────┬───────────────────┐
│ Country_of_Operator ┆ Purpose        ┆ Expected_Lifetime │
│ ---                 ┆ ---            ┆ ---               │
│ str                 ┆ str            ┆ f64               │
╞═════════════════════╪════════════════╪═══════════════════╡
│ USA                 ┆ Communications ┆ 7.554264          │
└─────────────────────┴────────────────┴───────────────────┘
```
