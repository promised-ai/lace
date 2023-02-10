# pylace

Python bindings to lace

## Install

Preliminaries Using pipenv:

```console
$ pipenv --three                             # create a virtual environment
$ pipenv shell                               # enter the virtual environment
$ pip install maturin pyarrow polars pandas  # install the build tool
```

To instll pylace

```console
$ maturin develop --release -m core/Cargo.toml  # install lace_core
$ pip install -e .                              # instal pylace
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

engine = lace.Engine.load('lace/resources/datasets/satellites/metadata.lace')

engine.predict('Class_of_Orbit', given={'Period_minutes': 1436.0})
# ('GEO', 0.13583714831550336)
```
