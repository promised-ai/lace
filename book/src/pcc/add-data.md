# Adding data to a model

Lace allow you to add and edit data without having to completely re-train.

You can edit existing cells,

<div class=tabbed-blocks>

```python
from lace.examples import Animals

animals = Animals()
animals.edit_datum(row='pig', column='fierce', value=0)
```

```rust,noplayground
use lace::examples::Example
use lace::prelude::*;

let mut animals = Example::Animals.engine().unwrap();

let write_mode = WriteMode::unrestricted();
let row = vec![Row {
    row_ix: String::from("pig"),
    values: vec![Value {
        col_ix: String::from("fierce"),
        value: Datum::Categorical(0),
    }],
}];

engine.insert_data(rows, None, None, write_mode).unwrap();
```

</div>

you can remove existing cells (set the value as missing),

<div class=tabbed-blocks>

```python
animals.edit_datum(row='otter', column='spotted', value=None)
```

```rust,noplayground
let write_mode = WriteMode::unrestricted();
let row = vec![Row {
    row_ix: String::from("otter"),
    values: vec![Value {
        col_ix: String::from("spotter"),
        value: Datum::Missing,
    }],
}];

engine.insert_data(rows, None, None, write_mode).unwrap();
```

</div>

and you can append new rows.

<div class=tabbed-blocks>

```python
animals.append_rows({
    'tribble': {'fierce': 1, 'meatteeth': 0, 'furry': 1},
})
```

```rust,noplayground
let write_mode = WriteMode::unrestricted();
let tribble = vec![Row {
    row_ix: String::from("tribble"),
    values: vec![
        Value {
            col_ix: String::from("fierce"),
            value: Datum::Categorical(1),
        },
        Value {
            col_ix: String::from("meatteeth"),
            value: Datum::Categorical(0),
        },
        Value {
            col_ix: String::from("furry"),
            value: Datum::Categorical(1),
        },
    ],
}];

engine.insert_data(tribble, None, None, write_mode).unwrap();
```

</div>
