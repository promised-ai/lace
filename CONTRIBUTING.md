# Contributing to braid

## General Guidelines

- Don't use getters and setters if it causes indirection in a performance heavy
    piece of code


## Error handling
As a general rule of thumb, if the user messed it up, return a `Result::Err`, if we messed it up, panic. There are exceptions:

- Indexing out of bounds should just panic
- In internal code like `braid_cc` where the user is expected to know what
    they're doing and where input validation and proper error handling would be
    a performance and development burden, just panic


## Naming

- For counts, use `n_things` instead of `nthings`.
- For indices, use `thing_ix` instead of `thingix`.
- No abbreviations in things that should be camel case

Good:

```rust
pub struct ColumnIndex {
    col_ix: usize
}
```

bad:

```rust
pub struct ColIndex{
    colix: usize
}
```
