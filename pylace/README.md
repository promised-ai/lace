# pylace

Python bindings to lace

## Install

### Install lates from PyPI
```console
$ python3 -m pip install pylace
```

### Install latest from GitHub
Building from source requires the Rust compiler (git it [here](https://rustup.rs/)).

```console
$ python3 -m pip install git+https://git@github.com/promised-ai/lace.git#egg=pylace&subdirectory=pylace
```

## Use

Note that the engine currently only supports loading from an existing metadata
file. The following lace functions are supported:

- rowsim
- depprob
- predict
- logp
- simulate
- append_rows
- update

```python
import lace

# The required files can be found here: https://github.com/promised-ai/lace/tree/master/pylace/lace/resources/datasets/satellites
engine = lace.Engine(data_source="data.csv", codebook="codebook.yaml")

# Train the model for 10_000 steps
engine.update(10_000)

# Predict the orbit's class based on the orbit's period.
engine.predict('Class_of_Orbit', given={'Period_minutes': 1436.0})
# ('GEO', 0.13583714831550336)
```
