<style>
    #surp-hist {
        aspect-ratio: 4/3;
    }
</style>
# Surprisal 

Surprisal is a method by which users may find surprising (go figure) data such
as outliers, anomalies, and errors.

## In information theory
In information theoretic terms, "surprisal" (also referred to as
*self-information*, *information content*, and potentially other things) is
simply the negative log likelihood.

\\[
s(x) = -\log p(x) \\
\\]

\\[
s(x|y) = -\log p(x|y)
\\]

## In Lace

In the Lace `Engine`, you have the option to call `engine.surprisal` and the option
the call `-engine.logp`. There are differences between these two calls:

`engine.surprisal` takes a column as the first argument and can take optional row
indices and values. `engine.surprisal` computes the information theoretic surprisal
of a value in a particular position in the Lace table. `engine.surprisal` considers
only existing values, or hypothetical values at specific positions in the
table.

`-engine.logp` considers hypothetical values only. We provide a set of inputs and
conditions and as 'how surprised would we be if we saw this?'

## Interpreting surprisal values

Surprisal is not normalized insofar as the likelihood is not normalized. For
discrete distributions, surprisal will always be positive, but for tight
continuous distributions that can have likelihoods greater than 1, surprisal
can be negative. Interpreting the raw surprisal values is simply a matter of
looking at which values are higher or lower and by how much.

Transformations may not be very valuable. The surprised distribution is usually
very far from capital 'N' Normal (Gaussian).

```python
import plotly.express as px
from lace.examples import Satellites

engine = Satellites()

surp = engine.surprisal('Period_minutes')

# plotly support for polars isn't currently great
fig = px.histogram(surp.to_pandas(), x='surprisal')
fig.show()
```

{{#include html/surp-hist.html}}

Lots of skew in this distribution. The satellites example is especially nasty
because there are a lot of extremes when we're talking about spacecraft.
