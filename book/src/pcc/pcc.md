# Probabilistic Cross Categorization

Lace is built on a Bayesian probabilistic model called *Probabilistic
Cross-Categorization* (PCC). PPC groups \\(m\\) columns into \\(1, ..., m\\)
*views*, and within each view, groups the \\(n\\) rows into \\(1, ..., n\\)
*categories*. PCC uses a non-parametric prior process (the [Dirichlet
process](https://en.wikipedia.org/wiki/Dirichlet_process)) to learn the number
of view and categories. Each column (feature) is then modeled as a [mixture
distribution](https://en.wikipedia.org/wiki/Mixture_model) defined by the
category partition. For example, a continuous-valued column will be modeled as
a mixture of Gaussian distributions. For references on PCC, see [the appendix](/glossary.md).

## Differences between PCC and Traditional ML

### Input Data

Most ML models are designed to handle one type of data, generally continuous.
This means if you have categorical data, you have to transform it: you can call
`float(x)` and just sweep the categorical-ness of the data under the rug, you
can do something like [one-hot
encoding](https://en.wikipedia.org/wiki/One-hot), which significantly increases 
dimensionality, or you can use some kind of embedding [like in
natural language processing](https://en.wikipedia.org/wiki/Word_embedding),
which destroys interpretability. PCC allows your data to stay as they are.

### The learning target

Most of the machine learning models we all know and love are formalized in
terms of learning some unknown function \\(f(x) \rightarrow y\\), where
\\(x\\)) are inputs and \\(y\\) are outputs, by optimizing with respect to some
objective (e.g. error; cross-entropy). This formalization locks machine
learning models into rigid use cases, e.g. predict "this" given "that". PCC
attempts to learn a joint probability distribution, \\(f(x_1, x_2, ...,
x_n)\\), where the \\(x\\)'s are features. From this joint distribution, the
user can construct any conditional distribution (e.g., \\(p(x_1, x_2 | x_3
)\\)). 

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
