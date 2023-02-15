# Surprisal


## In information theory
In information theoretic terms, "surprisal" is simply the negative log
likelihood.

\\[
s(x) = -\log p(x) \\
s(x|y) = -\log p(x|y)
\\]

## In Lace

In the Lace client, you have the option to call `c.surprisal` and the option
the call `-c.logp`. There are differences between these two calls:

`c.surprisal` takes a column as the first argument and can take option row
indies and values. `c.surprisal` computes the information theoretic surprisal
of a value in a particular position in the Lace table. `c.surprisal` considers
only existing values, or hypothetical values at specific positions in the
table. We as Redpoll 'how surprising is this observation'?

`-c.logp` considers hypothetical values only. We provide a set of inputs and conditions and as 'how surprised would we be if we saw this?'

## Interpreting surprisal values

Surprisal is not normalized insofar as the likelihood is not normalized. For
discrete distributions, surprisal will always be positive, but for tight
continuous distributions that can have likelihoods greater than 1, surprisal
can be negative. Interpreting the raw surprisal values is simply a matter of
looking at which values are larger or smaller and by how much.

There are a couple of options for transformation such as computing the standard
deviations from the minimum surprisal that a particular surprisal values is.
Or, knowing the minim surprisal, you could shift and scale the surprisal
distribution to go from [0, Inf).

\\[
from lace.examples import Animals

c = Animals()

col = 'Period_minutes'
\\]

The first thing we'll do is use `impute` to find the minimum possible
surprisal. Impute returns the most likely value for a cell. For each of the
imputed values, we compute the `surprisal`, which is the negative likelihood
value of the impute values in their specific cells in the table. Then we take
the `min`.

```python
imp = c.impute(col, rows=c.index, uncertainty_type=None)
imp_surps = c.surprisal(col, rows=imp.index, values=imp)
min_surp = imp_surps.surprisal.min()
```

Then we compute the surprisal of the rows in the table

```python
surp = c.surprisal(col, rows=imp.index)
```

Simple enough. We then estimate the standard deviation of surprisal values
using the surprisals of the observed values, and the min surprisals to make
some transformed values.

```python
stddev = surp.surprisal.std()
surp['shifted'] = surp.surprisal - min_surp
surp['scaled'] = (surp.surprisal - min_surp) / stddev

surp.sort_values(by='surprisal', ascending=False) \
    .head(15)
```

|                                                                                |   Period_minutes |   surprisal |   shifted |   scaled |
|:-------------------------------------------------------------------------------|-----------------:|------------:|----------:|---------:|
| Wind (International Solar-Terrestrial Program)                                 |         19700.5  |    12.2733  |   9.34141 | 10.412   |
| Integral (INTErnational Gamma-Ray Astrophysics Laboratory)                     |          4032.86 |     9.24784 |   6.31595 |  7.03978 |
| Chandra X-Ray Observatory (CXO)                                                |          3808.92 |     9.05307 |   6.12118 |  6.82268 |
| XMM Newton (High Throughput X-ray Spectroscopy Mission)                        |          2872.15 |     8.61971 |   5.68782 |  6.33966 |
| Geotail (Geomagnetic Tail Laboratory)                                          |          2474.83 |     8.39524 |   5.46335 |  6.08946 |
| RISat-2 (Radar Imaging Satellite 2)                                            |            41.2  |     7.76725 |   4.83536 |  5.3895  |
| Sirius 3 (SD Radio 3)                                                          |           994.83 |     7.29873 |   4.36684 |  4.86729 |
| Spektr-R/RadioAstron                                                           |             0.22 |     7.29629 |   4.3644  |  4.86457 |
| Interstellar Boundary EXplorer (IBEX)                                          |             0.22 |     7.29195 |   4.36006 |  4.85973 |
| Galileo IOV-1 FM2                                                              |           846.98 |     7.2889  |   4.35701 |  4.85634 |
| Galileo IOV-1 PFM                                                              |           846.88 |     7.28708 |   4.35519 |  4.8543  |
| Sirius 1 (SD Radio 1)                                                          |          1418.5  |     7.26576 |   4.33387 |  4.83054 |
| Galileo IOV-2 FM3                                                              |           844.76 |     7.24811 |   4.31622 |  4.81087 |
| Galileo IOV-2 FM4                                                              |           844.69 |     7.24682 |   4.31493 |  4.80942 |
| THEMIS E (Time History of Events and Macroscale Interactions during Substorms) |          1875.53 |     7.14463 |   4.21274 |  4.69553 |



