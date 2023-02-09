# Introduction

In this document we will introduce the required vocabulary of probabilistic ML, familiarize readers with the Probabilistic Cross-Categorization model, and provided a full overview of Lace including installation, data setup, training/fitting, and how to ask and answer different types of questions.

# Lace

Lace is an [*intuitive*](#) machine learning engine that **allows users to learn about their data** quickly by eliminating the post-training cost of asking answering questions. Lace facilitates the following machine learning capabilities

- Structure learning
- Row similarity / record relevance
- Likelihood computation
- Synthetic data and Simulation
- Imputation
- Prediction
- Uncertainty quantification
- Anomaly detection

Under the hood, Lace is a Probabilistic Cross-Categorization (PCC) engine. PCC is a Bayesian probabilistic model that learns a joint probability distribution over entire datasets. The full joint distribution can be broken down into conditional distribution that allow users to ask questions about any variable(s) given (optional) context provided by a set of any other variables.

## Useage

Lace can be used from python or rust, and there is a CLI to facilitate codebook generation and fitting.

The general workflow is as follows:

Generate a template codebook from your dataset, which you can edit

```console
$ lace codebook --csv mydata.csv mycodebook.yml
```

Run (train or fit) the lace model. In the below example, we run 32 states for 5000 iterations and save the model to `metadata.lace`

```console
$ lace run --csv mydata.csv --codebook mycodebook.yml -n 5000 -s 32 metadata.lace
```

Open the file in `pylace`

```python
import lace

engine = lace.Engine.load('metadata.lace')
```

## Examples
