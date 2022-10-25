# pybraid

Python bindings to braid

## Install

Preliminaries Using pipenv:

```console
$ pipenv --three        # create a virtual environment
$ pipenv shell          # enter the virtual environment
$ pip install maturin   # install the build tool
```

To build in dev mode

```console
$ maturin develop
```

## Use

Note that the engine currently only supports loading from an existing metadata
file. The following braid functions are supported:

- rowsim
- depprob
- predict
- logp
- simulate
- append_rows
- update

```python
import pybraid

engine = pybraid.Engine('animals.rp')

engine.predict('swims', given={'flippers': 1})
# (1, 0.13583714831550336)
```
