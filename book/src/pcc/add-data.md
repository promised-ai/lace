# Adding data to a model

Lace allow you to add and edit data without having to completely re-train.

You can edit existing cells,

<div class=tabbed-blocks>

```python
from lace.examples import Animals

animals = Animals()
animals.edit_cell(row='pig', col='fierce', value=0)

assert animals['pig', 'fierce'] == 0
```

```rust,noplayground
use lace::examples::Example;
use lace::prelude::*;

let mut animals = Example::Animals.engine().unwrap();

let write_mode = WriteMode::unrestricted();
let rows = vec![Row {
    row_ix: String::from("pig"),
    values: vec![Value {
        col_ix: String::from("fierce"),
        value: Datum::Categorical(lace::Category::U8(0)),
    }],
}];

animals.insert_data(rows, None, write_mode).unwrap();
```

</div>

you can remove existing cells (set the value as missing),

<div class=tabbed-blocks>

```python
animals.edit_cell(row='otter', col='brown', value=None)

assert animals['otter', 'brown'] is None
```

```rust,noplayground
let write_mode = WriteMode::unrestricted();
let rows = vec![Row {
    row_ix: String::from("otter"),
    values: vec![Value {
        col_ix: String::from("spots"),
        value: Datum::Missing,
    }],
}];

animals.insert_data(rows, None, write_mode).unwrap();
```

</div>

you can append new rows,

<div class=tabbed-blocks>

```python
animals.append_rows({
    'tribble': {'fierce': 1, 'meatteeth': 0, 'furry': 1},
})

assert animals['tribble', 'fierce'] == 1
```

```rust,noplayground
let write_mode = WriteMode::unrestricted();
let tribble = vec![Row {
    row_ix: String::from("tribble"),
    values: vec![
        Value {
            col_ix: String::from("fierce"),
            value: Datum::Categorical(lace::Category::U8(1)),
        },
        Value {
            col_ix: String::from("meatteeth"),
            value: Datum::Categorical(lace::Category::U8(0)),
        },
        Value {
            col_ix: String::from("furry"),
            value: Datum::Categorical(lace::Category::U8(1)),
        },
    ],
}];

animals.insert_data(tribble, None, write_mode).unwrap();
```

</div>

and you can even append new columns.

<div class=tabbed-blocks>

```python
cols = pd.DataFrame(
    pd.Series(["blue", "geen", "blue", "red"], name="fav_color"),
    index=["otter", "giant+panda", "dolphin", "bat"]
)

# lace will infer the column metadata, or you can pass the metadata in
animals.append_columns(cols)

assert animals["bat", "fav_color"] == "red"
```
</div>

Some times you may need to supply lace with metadata about the column.

<div class=tabbed-blocks>

```python
from lace import ColumnMetadata, ContinuousPrior


cols = pd.DataFrame(
    pd.Series([0.0, 0.1, 2.1, -1.3], name="fav_real_number")
    index=["otter", "giant+panda", "dolphin", "bat"]
)

md = ColMetadata.continuous(
    "fav_real_number", 
    prior=ContinuousPrior(0.0, 1.0, 1.0, 1.0)
)

animals.append_columns(cols, col_metadata={"fav_real_number", md})

assert animals["otter", "fav_real_number"] == 0.0
assert animals["antelope", "fav_real_number"] is None
```
</div>

Note that when you add a column you'll need to run inference (fit) a bit to
incorporate the new information.
