# Probabilistic Cross Categorization

Lace is built on a Bayesian probabilistic model called *Probabilistic
Cross-Categorization* (PCC). PPC groups \\(m\\) columns into \\(1, ..., m\\)
*views*, and within each view, groups the \\(n\\) rows into \\(1, ..., n\\)
*categories*. PCC uses a non-parametric prior process (the [Dirichlet
process](https://en.wikipedia.org/wiki/Dirichlet_process)) to learn the number
of view and categories. Each column (feature) is then modeled as a [mixture
distribution](https://en.wikipedia.org/wiki/Mixture_model) defined by the
category partition. For example, a continuous-valued column will be modeled as
a mixture of Gaussian distributions. For references on PCC, see [the
appendix](/appendix/references.md).

## Differences between PCC and Traditional ML

### Inputs and outputs

Regression and classification are defined in terms of learning a funciton
\\(f(x) \rightarrow y \\) that maps inputs, \\(x\\), to outputs, \\(y\\). PCC has
no notion of inputs and outputs. There is only data. PCC learns a joint
distribution \\(p(x_1, x_2, ..., x\_m)\\) from which the user can create condition
distributions. To predict \\(x_1\\) given \\(x_2\\) and \\(x_3\\), you find 
\\(\text{argmax}_{x_1} p(x_1|x_2, x_3)\\).

### Supported data types

Most ML models are designed to handle one type of input data, generally
continuous. This means if you have categorical data, you have to transform it:
you can convert it to a float (e.g. `float(x)` in python) and just
sweep the categorical-ness of the data under the rug, you can do something like
[one-hot encoding](https://en.wikipedia.org/wiki/One-hot), which significantly
increases dimensionality, or you can use some kind of embedding [like in
natural language processing](https://en.wikipedia.org/wiki/Word_embedding),
which destroys
interpretability. PCC allows your data to stay as they are.

### The learning method

Most machine learning models use an optimization algorithm to find a set of
parameters that achieves a local minima in the loss function. For example, Deep
Neural Networks may use stochastic gradient descent to minimize cross entropy.
This results in one parameter set representing one model.

In Lace, we use Markov Chain Monte Carlo to do posterior sampling. That is, we
attempt to draw a number of PCC states from the posterior distribution. These
states provide a kind of topographical map of the PCC posterior distribution
which we can use to do a number of things including computing likelihoods and
uncertainties.
