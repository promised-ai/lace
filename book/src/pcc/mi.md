# Mutual information

Mutual information (often referred to in code as `mi`) is a measure of the
information shared between two variables. Is is mathematically defined as

\\[
I(X;Y) = \sum_{x \in X} \sum_{y \in Y} p(x,y) \log \frac{p(x, y)}{p(x)p(y)},
\\]

or in terms of entropies,

\\[
I(X;Y) = H(X) - H(X|Y).
\\]

Mutual information is well behaved for discrete data types (count and
categorical), for which the sum applies; but for continuous data types for
which the sum becomes an integral, mutual information can break down because
differential entropy is no longer guaranteed to be positive.

For example, the following plots show the dependence probability and mutual
information heatmaps for the zoo dataset, which is composed entirely of binary
variables:

<div class=tabbed-blocks>

```python
from lace import examples

animals = examples.Animals()

animals.clustermap('depprob', color_continuous_scale='greys', zmin=0, zmax=1)
```
</div>

{{#include html/animals-depprob.html}}

**Above.** A [dependence probability](/pcc/depprob) cluster map for the Animals dataset. 

<div class=tabbed-blocks>

```python
animals.clustermap('mi', color_continuous_scale='greys')
```
</div>

{{#include html/animals-mi-unnormed.html}}

**Above.** A mutual information clustermap. Each cell represents the Mutual
Information between two columns. Note that compared to dependence probability,
the matrix is quite sparse. Also note that the diagonal entries are the
entropies for each column. 

And below are the dependence probability and mutual information heatmaps of the
satellites dataset, which is composed of a mixture of categorical and
continuous variables: 

<div class=tabbed-blocks>

```python
satellites = examples.Satellites()
satellites.clustermap('depprob', color_continuous_scale='greys', zmin=0, zmax=1)
```
</div>

{{#include html/sats-depprob.html}}

**Above.** The dependence probability cluster map for the satellites date set.

<div class=tabbed-blocks>

```python
satellites.clustermap('mi', color_continuous_scale='greys')
```
</div>

{{#include html/sats-mi-iqr.html}}

**Above.** The normalized mutual information cluster map for the satellites
date set. Note that the values are no longer bounded between 0 and 1 due to
inconsistencies caused by differential entropies.

<div class=tabbed-blocks>

```python
satellites.clustermap(
    'mi',
    color_continuous_scale='greys',
    fn_kwargs={'mi_type': 'linfoot'}
)
```
</div>

{{#include html/sats-mi-linfoot.html}}

**Above.** The Linfoot-transformed mutual information cluster map for the
satellites date set. The Linfoot information transformation often helps to
mediate the weirdness that can arise from differential entropy.

## Normalization Methods

Mutual information can be difficult to interpret because it does not have
well-behaved bounds. In all but the continuous case (in which the mutual
information could be negative), mutual information is only guaranteed to be
greater than zero. To create an upper bound, we have a number of options:

### Normalized

Knowing that the mutual information cannot exceed the minimum of the total
information in (the entropy of) either X or Y, we can normalize by the minimum
of the two component entropies:

\\[
\hat{I}(X;Y) = \frac{I(X; Y)}{\min \left[H(X), H(Y) \right]}
\\]

<div class=tabbed-blocks>

```python
animals.clustermap(
    'mi',
    color_continuous_scale='greys',
    fn_kwargs={'mi_type': 'normed'}
)
```
</div>

{{#include html/animals-mi-normed.html}}

**Above.** Normalized mutual information cluster map for the animals dataset.

### IQR

In the Information Quality Ratio (IQR), we normalize by the joint entropy.

\\[
\hat{I}(X;Y) = \frac{I(X; Y)}{H(X, Y)}
\\]

<div class=tabbed-blocks>

```python
animals.clustermap(
    'mi',
    color_continuous_scale='greys',
    fn_kwargs={'mi_type': 'iqr'}
)
```
</div>

{{#include html/animals-mi-iqr.html}}

**Above.** IQR Normalized mutual information cluster map for the animals
dataset.

### Jaccard

To compute the Jaccard distance, we subtract the IQR from 1. Thus, columns with
more shared information have smaller distance

\\[
\hat{I}(X;Y) = 1 - \frac{I(X; Y)}{H(X, Y)}
\\]

<div class=tabbed-blocks>

```python
animals.clustermap(
    'mi',
    color_continuous_scale='greys',
    fn_kwargs={'mi_type': 'jaccard'}
)
```
</div>

{{#include html/animals-mi-jaccard.html}}

**Above.** Jaccard distance cluster map for the animals dataset.

### Pearson

To compute something akin to the Pearson Correlation coefficient, we normalize
by the square root of the product of the component entropies:

\\[
\hat{I}(X;Y) = \frac{I(X; Y)}{\sqrt{H(X) H(Y)}}
\\]

<div class=tabbed-blocks>

```python
animals.clustermap(
    'mi',
    color_continuous_scale='greys',
    fn_kwargs={'mi_type': 'pearson'}
)
```
</div>

{{#include html/animals-mi-pearson.html}}

**Above.** Pearson normalized mutual information matrix cluster map for the
animals dataset.

### Linfoot

Linfoot information is the solution to solving for the correlation between the
X and Y components of a bivariate Gaussian distribution with given mutual
information.

\\[
\hat{I}(X;Y) = \sqrt{ 1 - \exp(2 - I(X;Y)) }
\\]

<div class=tabbed-blocks>

```python
animals.clustermap(
    'mi',
    color_continuous_scale='greys',
    fn_kwargs={'mi_type': 'linfoot'}
)
```
</div>

Linfoot is often the most well-behaved normalization method especially when
using continuous variables.

{{#include html/animals-mi-linfoot.html}}

**Above.** Linfoot information matrix cluster map for the animals dataset.

### Variation of Information

The Variation of Information (VOI) is a metric typically used to determine the
distance between two clustering of variables, but we can use it generally to
transform mutual information into a valid metric.

\\[
\text{VI}(X;Y) = H(X) + H(Y) - 2 I(X,Y)
\\]

<div class=tabbed-blocks>

```python
animals.clustermap(
    'mi',
    color_continuous_scale='greys',
    fn_kwargs={'mi_type': 'voi'}
)
```
</div>

{{#include html/animals-mi-voi.html}}

**Above.** Variation of information matrix.

