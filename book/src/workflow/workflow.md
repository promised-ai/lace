# The lace workflow

The typical workflow consists of two or three steps:

1. [Create a codebook](codebook.md)
2. [Run/fit/train a model](model.md)
3. [Ask questions](analysis.md)

Step 1 is optional in many cases as Lace usually does a good job of inferring
the types of your data. The condensed workflow looks like this.


Create an optional codebook using the CLI.

```console
$ lace codebook --csv data.csv codebook.yaml
```

Run a model.

```console
$ lace run --csv data.csv --codebook codebook.yaml -n 5000 metadata.lace
```

Open the model in lace

<div class=tabbed-blocks>

```python
import lace

engine = lace.Engine(metadata='metadata.lace')
```

```rust
use lace::Engine;

let engine = Engine::load("metadata.lace")?;
```
</div>
