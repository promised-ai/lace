# Nomenclature

Here we list some of the terminology you will encounter when using lace. We
also attempt to provide a bit of basics on the Bayesian paradigm for users who
are more accustomed to traditional machine learning tools.

## Glossary

- **view**: A cluster of columns within a state.
- **category**: A cluster of rows within a view.
- **state**: what we call a lace posterior sample. Each state represents the
    current configuration (or state) of an independent markov chain. We
    aggregate over states to achieve estimate of likelihoods and uncertainties.
- **metadata**: A set of files from which an `Engine` may be loaded
- **prior**: A probability distribution that describe how likely certain
    hypotheses (model parameters) are before we observe any data.
- **hyperprior**: A prior distribution on prior. Allows us to admit a larger
    amount of initial uncertainty and permit a broader set of hypotheses.
- **empirical hyperprior**: A hyperprior with parameters derived from data.

## What you need to know about Bayesian Statistics

Bayesian statistics is built around the idea of *posterior inference*. The
*posterior distribution* is the probability distribution of the parameters,
\\(\theta\\), of some model given observed data, \\(x\\). In math: \\( p(\theta
| s) \\). Per Bayes theorem, the posterior distribution can be written in terms
of other distributions,

\\[
p(\theta | s) = \frac{p(x|\theta)p(\theta)}{p(x)}.
\\]

Where \\( p(x|\theta) \\) is the *likelihood* of the observations given the
parameters of our model; \\( p(\theta) \\) is the *prior distribution*, which
defines our beliefs about the model parameters in the absence of data; and \\(
p(x) \\) is the *marginal likelihood*, which the likelihood of the data
marginalized over all possible models. Of these, the likelihood and prior are
the two distributions we're most concerned with. The marginal likelihood, which
is defined as

\\[
    p(x) = \int p(x|\theta)p(\theta) d\theta
\\]

is notoriously difficult and a lot of effort in Bayesian computation goes
toward making the marginal likelihood go away, so we won't talk about it much.

## Dirichlet process

The dirichlet pr
