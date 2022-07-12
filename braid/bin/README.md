# Regression tests

The regression command takes the path to the yaml config and the path of the json output.

```
$ braid regression <YAML_IN> <JSON_OUT>
```

The input in yaml because yaml is easier to read and edit; the output is JSON because there are much faster libraries for deserializing JSON in python. Pyyaml was becoming a bottleneck especially with large outputs.

## The Main Configuration

The configuration is a yaml file with fields corresponding to configurations of the individual tests you want to run.

- id: A string name for the config. Used by the report generator to group configs and plot their results over time.
- pit (optional): The test that measures how well braid fits against real data by using the Probability Integral Transform (PIT).
- geweke (optional): Tests whether the MCMC sampler is sampling from the correct posterior distribution.
- shapes (optional): Tests how well braid fits against known zero-correlations data sets at varying scales.
- benchmark (optional): Runs timed benchmarks on states of various sizes and structure.

## The PIT test

The `pit` test does inference on a dataset and then measures how well each column's distribution matches the empirical distribution of the data. The term `pit` is misleading because error is computed differently for different data types:

- Continuous: [PIT](https://en.wikipedia.org/wiki/Probability_integral_transform)
- Categorical: CDF error

We do not expect the fit to always be perfect. There are many factors that could cause a poor fit to an individual column. For example if column X is related to column Y, X and Y will be in the same view, but if X is multimodal abd Y is not, we might have trouble fitting to X. In this case, there is likely a confounding variable that we do not have access to. If we did fit well in this case, we might be *overfit* and risk overconfidence and production failure.

### Configuration

The configuration contains the following fields:

- datasets: a list of datasets with the number of states (n_states) and the number of iterations (n_iters) to run the dataset. Valid datasets are currently `animals`, `satellites`, and `satellites-normed`. Any dataset can be used if the following directory structure is present in the directory from which the regression is run:

```
resources
+-- datasets
    +-- <my_dataset>
        +-- data.csv
        +-- codebook.yaml
```

### Result

The result is a map of string dataset names to vectors of `FeatureErrorResult`. The `FeatureErrorResult` are ordered by their column order.

### Example PIT configuration

```yaml
- animals
  nstates: 8
  n_iters: 1000
- satellites
  nstates: 8
  n_iters: 1000
```

## The Geweke test

The Geweke test is a scaleable way to test whether an MCMC sampler draws from the correct distribution. It is very straight forward. 

### How it works 

First you sample the *prior chain* by drawing from joint distribution of data and parameters, $f(\theta, x)$ many times by initializing a data-less state, $\theta$, then drawing from $x$ from $f(x|\theta)$. Then you sample from the *posterior chain* by initializing one sample from the prior chain then repeatedly updating the parameters and data by cycling through $\theta' \sim f(\theta|x)$ (the MCMC step) and $x' \sim f(x|\theta')$. If the MCMC kernel is correct, both methods should draw from the same distribution.

### Configuration

The config has the following fields:
- settings: A map from config name to `StateGewekeSettings`. Each config will be run.
- n_runs: The number of chain sets to run. Each chain set is one prior chain and one posterior chain.
- n_iters: The number of iterations/samples per chain set.
- lag (optional): The number of samples to ignore between collection of posterior chain samples. This helps reduce the autocorrelation in the samples.

**Note:** The states should be small because the sampler will not be able to span the joint space in a reasonable amount of time for large states.

```yaml
n_iters: 1000
n_runs: 2
lag: 1
settings:
  # Run one geweke test usin the finite_cpu algorithm on an all-continuous
  # state.
  finite_cpu:
    ncols: 4
    nrows: 50
    row_alg: finite_cpu
    col_alg: finite_cpu
    transitions:
      - column_assignment
      - row_assignment
      - state_alpha
      - view_alphas
      - feature_priors
    cm_types:
      - continuous
      - continuous
      - continuous
      - continuous
```

### Results

- results: A map from field (String) to `GewekeResult`
- aucs: The area between the curve and the 1-1 line comparing the empirical CDF from both the prior and posterior chains.

## The Shapes test

The shapes test creates a bunch of zero-correlation 2-D shapes, fits an `Engine` to them and then determines whether the inferred shape is close to the true shape using a permutation test. The shape options are `ring`, `wave`, `square`, `x`, and `dots`.

Each shape is run small and scaled to large magnitude. This is to check whether the hyper prior or numerical issues affect inference at large magnitude. If so, it may be necessary to scale columns.

### Configuration

The config has the following fields:

- shapes: a list of shapes to test.
- n: the number of data to generate from each shape.
- n_perms: the number of permutations to use for the permutation tests. More means more accuracy, but the permutation test tends to be very slow because it is $O(n^2)$.
- n_states (optional): The number of states to use in inference (default: 8).

```yaml
shapes:
  - ring
  - wave
  - square
  - x
  - dots
n: 500
n_perms: 500
```

## Result

The result contains a vec of shape results which contains results for a scaled and unscaled shape run. The individual run contains the following fields.

- shape: ShapeType,
- n: usize, the number of samples
- n_perms: usize, the numbre of permutations for the test
- p: f64, the p value of the permutation test (we want higher)
- observed: Vec<Vec<f64>>, the input samples from the shape
- simulated: Vec<Vec<f64>>, the output samples from the engine

## Benchmarks

The config has the following fields:

- ncats: A list of the number of categories per view states may have
- nviews: A list of the number of views states may have
- nrows: A list of the number of rows states may have
- ncols: A list of the number of columns states may have
- row_algs: A list of the row algorithms to run (`gibbs`, `finite_cpu`, `slice`)
- col_algs: A list of the column algorithms to run (`gibbs`, `finite_cpu`, `slice`)
- n_runs: The number of replicate runs.

The Cartesian product of all lists will be run `n_runs` times, so minimize your lists.

### Example benchmark config

The following configuration will run 16 benchmarks 5 times each.

```yaml
ncats:
  - 1
  - 2
nviews:
 - 1
 - 2
nrows:
  - 10
  - 100
ncols:
  - 2
  - 10
row_algs:
  - finite_cpu
col_algs:
  - finite_cpu
n_runs: 5
```

### Result

- ncats: Vec<usize>, the number of categories
- nviews: Vec<usize>, the number of views
- nrows: Vec<usize>, the number of rows
- ncols: Vec<usize>, the number of columns
- row_asgn_alg: Vec<RowAssignAlg>, the row reassignment algorithm used
- col_asgn_alg: Vec<ColAssignAlg>, the column reassignment algorithm used
- run: Vec<usize>, the run replicate id
- time_sec: Vec<f64>, the run time in seconds

The index `i` of each vector corresponds to that variable on the `i`th run.

# Example configuration

```yaml
# Quick running smoke-test version of the regression config.
---
id: "quick"
pit:
  datasets:
    - satellites-normed:
        nstates: 8
        n_iters: 1000
    - satellites:
        nstates: 8
        n_iters: 1000
    - animals:
        nstates: 8
        n_iters: 1000
shapes:
  shapes:
    - ring
    - wave
    - x
  n: 100
  n_perms: 100
geweke:
  n_iters: 1000
  n_runs: 2
  lag: 1
  settings:
    finite_cpu:
      ncols: 4
      nrows: 50
      row_alg: finite_cpu
      col_alg: finite_cpu
      transitions:
        - column_assignment
        - row_assignment
        - state_alpha
        - view_alphas
        - feature_priors
      cm_types:
        - continuous
        - continuous
        - continuous
        - continuous
benchmark:
  ncats:
    - 1
    - 2
  nviews:
   - 1
   - 2
  nrows:
    - 10
    - 100
  ncols:
    - 2
    - 10
  row_algs:
    - finite_cpu
  col_algs:
    - finite_cpu
  n_runs: 5
```
