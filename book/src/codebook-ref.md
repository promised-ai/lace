# Codebook reference

The codebook is how you tell Lace about your data. The codebook contains
information about

- Row names
- Column names
- The type of data in each column (e.g., continuous, categorical, or count)
- The prior on the parameters for each column
- The hyperprior on the prior parameters for each column
- The prior on the Dirichlet Process alpha parameter

## Codebook fields

### `table_name`

String name of the table. For your reference.

### `state_prior_process`

The prior process used for assigning columns to views. Can either be a Dirichlet process with a Gamma prior on alpha:

```yaml,deserializeTo=StatePriorProcess
state_prior_process: !dirichlet
  alpha_prior:
    shape: 1.0
    rate: 1.0
```

or a Pitman-Yor process with a Gamma prior on alpha and a Beta prior on d.

```yaml,deserializeTo=StatePriorProcess
state_prior_process: !pitman_yor
  alpha_prior:
    shape: 1.0
    rate: 1.0
  d_prior:
    alpha: 0.5
    beta: 0.5
```

### `view_prior_process`

The prior process used for assigning rows to categories. Can either be a Dirichlet process with a Gamma prior on alpha:

```yaml,deserializeTo=ViewPriorProcess
view_prior_process: !dirichlet
  alpha_prior:
    shape: 1.0
    rate: 1.0
```

or a Pitman-Yor process with a Gamma prior on alpha and a Beta prior on d.

```yaml,deserializeTo=ViewPriorProcess
view_prior_process: !pitman_yor
  alpha_prior:
    shape: 1.0
    rate: 1.0
  d_prior:
    alpha: 0.5
    beta: 0.5
```

### `col_metadata`

A list of columns, ordered by left-to-right occurrence in the data. Contains
the following fields:

- `name`: The name of the column
- `notes`: Optional information about the column. Purely for reference
- `coltype`: Contains information about the type type of data, the prior, and
    the hyper prior. See [column metadata for more
    information](/basics/codebook#column-metadata)
- `missing_not_at_random`: a boolean. If `false` (default), missing values in
    the column are assumed to be missing completely at random.

### `row_names`
A list of row names in order of top-to-bottom occurrence in the data

### `notes`
Optional notes for user reference

## Codebook type inference

When you upload your data, Lace will pull the row and column names from the
file, infer the data types, and choose and empirical hyperprior from the data.

Type inference works like this:

- Categorical if:
    + The column contains only string values
        * Lace will assume the categorical variable can take on any of (and
            only) the existing values in the column
    + There are only integers up to a cutoff.
        * If There are only integers in the column `x` the categorical values
            will be assumed to take on values 0 to `max(x)`.
- Count if:
    + There are only integers that exceed the value of the cutoff
- Continuous if:
    + There are only integers and one or more floats

## Column metadata

- Either `prior` or `hyper` must be defined.
    + If `prior` is defined and `hyper` is not defined, hyperpriors and
        hyperparameter inference will be disabled.

<p class=warning>
It is best to leave the hyperpriors alone. It is difficult to intuit what
effect the hyperpriors have on the final distribution. If you have knowledge
beyond the vague hyperpriors, null out the `hyper` field with a `~` and set the
prior instead. This will disable hyperparameter inference inf favor of the
expert knowledge you have provided.
</p>

### Continuous

The continuous type has the `hyper` field and the `prior` field. The prior
parameters are those for the Normal Inverse Chi-squared prior on the mean and
variance of a normal distribution.

- `m`: the prior mean
- `k`: how strongly (in pseudo observations) that we believe `m`
- `s2`: the prior variance
- `v`: how strongly (is pseudo observations) that we believe `s2`

To have widely dispersed components with small variances you would set `k` very
low and very high.

FIXME: Animation showing effect of different priors

The hyper priors are the priors on the above parameters. They are named for the
parameters to which they are attached, e.g. `pr_m` is the hyper prior for the
`m` parameter.

- `pr_m`: Normal distribution
- `pr_k`: Gamma distribution with shape and rate (inverse scale) parameters
- `pr_v`: Inverse gamma distribution with shape and scale parameters
- `pr_s2`: Inverse gamma distribution with shape and scale parameters

```yaml,deserializeTo=lace_codebook::ColMetadataList
- name: Eccentricity
  coltype: !Continuous
    hyper:
      pr_m:
        mu: 0.02465318142734303
        sigma: 0.1262297091840037
      pr_k:
        shape: 1.0
        rate: 1.0
      pr_v:
        shape: 7.0587581525186648
        scale: 7.0587581525186648
      pr_s2:
        shape: 7.0587581525186648
        scale: 0.015933939480678149
    prior:
      m: 0.0
      k: 1.0
      s2: 7.0
      v: 1.0
    # To not define the prior add a `~`
    # prior: ~
  notes: ~
  missing_not_at_random: false
```

### Categorical

In addition to `prior` and `hyper`, Categorical has additional special fields:

- `k`: the number of values the variable can assume
- `value_map`: An optional map of integers in [0, ..., k-1] mapping the integer
    code (how the value is represented internally) to the string value. If
    `value_map` is not defined, it is usually assume that classes take on only
    integer values in [0, ..., k-1].

The `hyper` is an inverse gamma prior on the prior parameter `alpha`

```yaml,deserializeTo=lace_codebook::ColMetadataList
- name: Class_of_Orbit
  coltype: !Categorical
    k: 4
    hyper:
      pr_alpha:
        shape: 1.0
        scale: 1.0
    value_map: !string
      0: Elliptical
      1: GEO
      2: LEO
      3: MEO
    prior:
      alpha: 0.5
      k: 4
    # To not define the prior add a `~`
    # prior: ~
  notes: ~
  missing_not_at_random: false
```

## Editing the codebook

You should use the default codebook generated by the Lace CLI as a starting
point for custom edits. Generally the only edits you will make are

- Adding notes/comments
- changing the `state_alpha_prior` and `view_alpha_prior` (though you should
    only do this if you know what you're doing)
- converting a `Count` column to a `Categorical` column. Usually there will be
    no need to change between other column types.
