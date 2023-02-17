# Create and edit a codebook

The codebook contains information about your data such as the row and column
names, the types of data in each column, how those data should be modeled, and
all the prior distributions on various parameters.

## The default codebook

In the lace CLI, you have the ability to initialize and run a model without
specifying a codebook.

```console
$ lace run --csv data -n 5000 metadata.lace
```

Behind the scenes, lace create a default codebook by inferring the types of
your columns and creating a very broad (but not quite broad enough to satisfy
the frequentists) hyper prior, which is a prior on the prior.

## Creating a template codebook

Lace is happy to generate a default codebook for you when you initialize a
model. You can create the default codebook using the CLI. To create a codebook
from a CSV file:

```console
$ lace codebook --csv data.csv codebook.yaml
```

If you use a data format with a schema, such as Parquet or IPC (feather v2),
you make Lace's work a bit easier.

```console
$ lace codebook --ipc data.feather codebook.yaml
```

If you want to make changes to the codebook, the most common of which are
editing the Dirichlet process prior, specifying whether certain columns are
missing not-at-random, adjusting the prior distributions and disabling hyper
priors.
