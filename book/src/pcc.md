# Probabilistic Cross Categorization

Lace is built on PCC, which is a Bayesian probabilistic model. In terms of functionality and use, PCC sits somewhere between methods like Deep Learning and Random Forests, and Probabilistic programming languages. Like random forests, PCC automates model creation

### Input Data

### The learning target

Most of the machine learning models we all know and love are formalized in terms of learning some unknown function \\(f(x) \rightarrow y\\), where \\(x\\)) are inputs and \\(y\\) are outputs. PCC, attempts to learn a joint probability distribution, \\(f(x_1, x_2, ..., x_n)\\), where the \\(x\\)'s are features. From this joint distribution, the user can construct any conditional distribution (e.g., \\(p(x_1, x_2 | x_3 )\\)). 

### The learning method

Most machine learning models use an optimization algorithm to find a set of parameters that achieves a local minia in the loss function. For example, Deep Neural Networks may use stochastic gradient descent to minimize cross entropy. This results in one parameter set representing one model.

In Lace, we use Markov Chain Monte Carlo to do posterior sampling. That is, we attempt to draw a number of PCC states from the posterior distribution. These states provide a kind of topographical map of the PCC posterior distribution which we can use to do a number of things including computing likelihoods and uncertainties.

## What is Probabilistic Cross-Categorization
