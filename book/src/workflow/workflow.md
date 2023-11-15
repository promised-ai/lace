# The lace workflow

The typical workflow consists of two or three steps:

1. [Create a codebook](codebook.md)
2. [Run/fit/train a model](model.md)
3. [Ask questions](analysis.md)

Step 1 is optional in many cases as Lace usually does a good job of inferring
the types of your data. The condensed workflow looks like this.

<div class=tabbed-blocks>

```python
import pandas as pd
import lace

df = pd.read_csv("mydata.csv", index_col=0)

# 1. Create a codebook (optional)
codebook = lace.Codebook.from_df(df)

# 2. Initialize a new Engine from the prior. If no codebook is provided, a
# default will be generated
engine = lace.Engine.from_df(df, codebook=codebook)

# 3. Run inference
engine.run(5000)
```

```rust,noplayground
use polars::prelude::{SerReader, CsvReader};
use lace::prelude::*;

let df = CsvReader::from_path("mydata.csv")
  .unwrap()
  .has_header(true)
  .finish()
  .unwrap();

// 1. Create a codebook (optional)
let codebook = Codebook::from_df(&df, None, None, False).unwrap();

// 2. Build an engine
let mut engine = EngineBuilder::new(DataSource::Polars(df))
  .with_codebook(codebook)
  .build()
  .unwrap();

// 3. Run inference
// Use `run` to fit with the default transition set and update handlers; use
// `update` for more control.
engine.run(5_000);
```

</div>

You can also use the CLI to create codebooks and run inference. Creating a default YAML codebook with the CLI, and then manually editing is good way to fine tune models.

```console
$ lace codebook --csv mydata.csv codebook.yaml
$ lace run --csv data.csv --codebook codebook.yaml -n 5000 metadata.lace
```
