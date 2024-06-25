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

Behind the scenes, lace creates a default codebook by inferring the types of
your columns and creating a very broad (but not quite broad enough to satisfy
the frequentists) hyper prior, which is a prior on the prior.

We can also create the default codebook in code.

<div class=tabbed-blocks>

```python
import polars as pl
from lace import Codebook
from lace.examples import ExamplePaths

# Here we get the path to an example csv file, but you can use any file that
# can be read into a polars or pandas dataframe
path = ExamplePaths("satellites").data
df = pl.read_csv(path)

# Infer the default codebook for df
codebook = Codebook.from_df("satellites", df)
```

```rust,noplayground
use polars::prelude::{CsvReader, SerReader};
use lace::codebook::Codebook;
use lace::examples::Example;

// Load an example file
let paths = Example::Satellites.paths().unwrap();
let df = CsvReader::from_path(paths.data)
    .unwrap()
    .has_header(true)
    .finish()
    .unwrap();

// Create the default codebook
let codebook = Codebook::from_df(&df, None, None, None, false).unwrap();
```

</div>

## Creating a template codebook

Lace is happy to generate a default codebook for you when you initialize a
model. You can create and save the default codebook to a file using the CLI. To
create a codebook from a CSV file:

```console
$ lace codebook --csv data.csv codebook.yaml
```

Note that if you love quotes and brackets, and hate being able to comment, you can use json for
the codebook. Just give the output of `codebook` a `.json` extension.

```console
$ lace codebook --csv data.csv codebook.json
```

If you use a data format with a schema, such as Parquet or IPC (Apache Arrow v2),
you make Lace's work a bit easier.

```console
$ lace codebook --ipc data.arrow codebook.yaml
```

If you want to make changes to the codebook -- the most common of which are
editing the Dirichlet process prior, specifying whether certain columns are
missing not-at-random, adjusting the prior distributions and disabling hyper
priors -- you just open it up in your text editor and get to work.

For example, let's say we wanted to make a column of the satellites data set
missing not-at-random, we first create the template codebook,

```console
$ lace codebook --csv satellites.csv codebook-sats.yaml
```

open it up in a text editor and find the column of interest

<div class=tabbed-blocks>

```yaml,deserializeTo=lace_codebook::ColMetadataList
- name: longitude_radians_of_geo
  coltype: !Continuous
    hyper:
      pr_m:
        mu: 0.21544247097911842
        sigma: 1.570659039531299
      pr_k:
        shape: 1.0
        rate: 1.0
      pr_v:
        shape: 6.066108090103747
        scale: 6.066108090103747
      pr_s2:
        shape: 6.066108090103747
        scale: 2.4669698184613824
    prior: null
  notes: null
  missing_not_at_random: false
```

```json,deserializeTo=lace_codebook::ColMetadata
{
  "name": "longitude_radians_of_geo",
  "coltype": {
    "Continuous": {
      "hyper": {
        "pr_m": {
          "mu": 0.21544247097911842,
          "sigma": 1.570659039531299
        },
        "pr_k": {
          "shape": 1.0,
          "rate": 1.0
        },
        "pr_v": {
          "shape": 6.066108090103747,
          "scale": 6.066108090103747
        },
        "pr_s2": {
          "shape": 6.066108090103747,
          "scale": 2.4669698184613824
        }
      },
      "prior": null
    }
  },
  "notes": null,
  "missing_not_at_random": false
}
```
</div>

and change the column metadata to something like this:

<div class=tabbed-blocks>

```yaml,deserializeTo=lace_codebook::ColMetadataList
- name: longitude_radians_of_geo
  coltype: !Continuous
    hyper:
      pr_m:
        mu: 0.21544247097911842
        sigma: 1.570659039531299
      pr_k:
        shape: 1.0
        rate: 1.0
      pr_v:
        shape: 6.066108090103747
        scale: 6.066108090103747
      pr_s2:
        shape: 6.066108090103747
        scale: 2.4669698184613824
    prior: null
  notes: "This value is only defined for GEO satellites"
  missing_not_at_random: true
```

```json,deserializeTo=lace_codebook::ColMetadata
{
  "name": "longitude_radians_of_geo",
  "coltype": {
    "Continuous": {
      "hyper": {
        "pr_m": {
          "mu": 0.21544247097911842,
          "sigma": 1.570659039531299
        },
        "pr_k": {
          "shape": 1.0,
          "rate": 1.0
        },
        "pr_v": {
          "shape": 6.066108090103747,
          "scale": 6.066108090103747
        },
        "pr_s2": {
          "shape": 6.066108090103747,
          "scale": 2.4669698184613824
        }
      },
      "prior": null
    }
  },
  "notes": null,
  "missing_not_at_random": true
}
```
</div>

Sometimes, we have a bit of knowledge that we can transfer to lace in the form
of a more-specific prior distribution. To set the prior we remove the hyper
prior and set the prior. Note that doing this disabled prior parameter
inference.

<div class=tabbed-blocks>

```yaml,deserializeTo=lace_codebook::ColMetadataList
- name: longitude_radians_of_geo
  coltype: !Continuous
    hyper: null
    prior: 
        m: 0.0
        k: 1.0
        v: 1.0
        s2: 3.0
  notes: "This value is only defined for GEO satellites"
  missing_not_at_random: true
```

```json,deserializeTo=lace_codebook::ColMetadata
{
  "name": "longitude_radians_of_geo",
  "coltype": {
    "Continuous": {
      "hyper": null,
      "prior": {
        "m": 0.0,
        "k": 1.0,
        "v": 1.0,
        "s2": 3.0
      }
    }
  },
  "notes": null,
  "missing_not_at_random": true
}
```
</div>

For a complete list of codebook fields, see [the reference](/codebook-ref.md).
