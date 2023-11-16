# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Lace can now be compiled to Wasm

### Changed

- Moved CLI into its own crate
- Moved `DataSource` variants for Parquet, IPC (Arrow), and JSON data types into the `formats` feature flag.
- Moved the `CtrlC` `UpdateHandler` into `ctrlc_handler` feature flag
- Moved `Bencher` into the `bencher` feature flag
- Moved `Example` code into the `examples` feature flag (on by default)
- Replaced instances of `once_cell::sync::OnceCell` with `syd::sync::OnceLock`
- Renamed all files/methods with the name `feather` to `arrow`
- Renamed `Builder` to `EngineBuilder`

### Fixed

- Fixed typo `UpdateHandler::finialize` is now `UpdateHandler::finalize`

## [python-0.4.1] - 2023-10-19

### Fixed

- Fixed issues when indexing `Engine` with slices.

## [rust-0.4.1] - 2023-10-16

### Fixed

- Fixed stuck iterator from `ValueMap.iter`.

## [python-0.4.0] - 2023-09-27

### Added
- Component params now available from `pylace`
- `Engine`s in `pylace` now implement deepcopy

### Fixed

- Fixed `pylace`'s `__getitem__` index name to match other accessors
- Fixed `pylace`'s `append_column_metadata` to refer to the correct internal function.
- `pylace` will try to use Python's internal conversion for floats in `value_to_datum` conversion. 

### Changed

- Updated rust `polars` to version `0.33`
- Updated rust `arrow2` to `0.18`

## [rust-0.4.0] - 2023-09-27

### Fixed

 - Upgraded `polars` version to version `0.33` (fixes builds failing due to no `cmake` installed)

## [python-0.3.1] - 2023-08-28

### Fixed

- Automatic conversion from `np.int*` types to Categorical.
- Updated lace version to 0.3.1

## [rust-0.3.1] - 2023-08-28

### Fixed

- Locked to a specific version of `polars 0.32` to produce reproducible builds on Rust 1.72

## [python-0.3.0] - 2023-07-25

### Changed

- Updated lace version to 0.3.0
- Updated polars to version ^0.31

## [rust-0.3.0] - 2023-07-25

### Fixed

- Updated breaking dependencies

## [python-0.2.0] - 2023-07-14

### Changed

- `lace.Engine.__init__` now takes a Pandas' or Polars' dataframe to initialize an `Engine`

### Added

- `Engine.load` supports loading metadata from disk
- New `Codebook` class in the `lace.codebook` module
- `lace.CodebookBuilder` supports loading a codebook from disk or using parameterized inference for creating a new `Engine`
- Can append new columns to an `Engine.append_columns`
    + Specify the types of new columns with `ColumnMetadata`
- Can delete columns using `Engine.del_column`
- New plot: `lace.plot.prediction_uncertainty`

## [rust-0.2.0] - 2023-07-14

### Changed

- `FTypeCompat` class now has `Debug` implemented
- `ColMetaDataList` now has `&str` indexing (i.e., traits `Index<&str>` and `IndexMut<&str>` are implemented)
- `Engine::insert_data` no longer takes `suppl_metadata` paraemter
- `SupportExtension` class now holds a `ValueMapExtension` rather than the `k_orig` and `k_ext` fields
- Certain variants of the `lace::interface::engine::error::InsertDataError` enum had typos fixed

### Removed

- The `NoOp` update handler is gone in favor of `()`

### Added

- `DataSource::Polars` to support direct loading of Polars' DataFrame
- `Engine::del_column` to delete columns
- `Codebook::from_df` to create a codebook from a polars DataFrame
- `()` now implements `UpdateHandler` (replaces `NoOp`)
- Exposed `ExamplePaths` to public API
- New method: `lace::codebook::data::series_to_colmd`
- New method: `CategoryMap::add`
- New method: `ValueMap::extend`
- New public types in `codebook::valuemap`: `ValueMapExtension`, `ValueMapExtensionError`

### Fixed

- Clippy Lint in `lace-stats`
- Fixed bug in `StateTimeout` update handler

## [python-0.1.2] - 2023-06-09

### Fixed

- `Engine.append_rows` would incorrectly raise an exception stating that
    duplicate indices were found in polars DataFrame

## [python-0.1.1] - 2023-06-07

### Added
- Allow index column name to be a case-insensitive variant of `ID` or `Index`
    across `Engine` methods

## [rust-0.1.2] - 2023-06-07

### Added
- Allow index column name to be a case-insensitive variant of `ID` or `Index`

## [rust-0.1.1] - 2023-05-31

### Fixed

- Documentation fixes
- Benchmark tests now work properly

## [python-0.1.0] - 2023-04-24

### Added

Initial release on [PyPi](https://pypi.org/)

## [rust-0.1.0] - 2023-04-20

### Added

Initial release on [crates.io](https://crates.io/)

[unreleased]: https://github.com/promised-ai/lace/compare/python-0.4.1...HEAD
[python-0.4.1]: https://github.com/promised-ai/lace/compare/python-0.4.0...python-0.4.1
[rust-0.4.1]: https://github.com/promised-ai/lace/compare/rust-0.4.0...rust-0.4.1
[python-0.4.0]: https://github.com/promised-ai/lace/compare/python-0.3.1...python-0.4.0
[rust-0.4.0]: https://github.com/promised-ai/lace/compare/rust-0.3.1...rust-0.4.0
[python-0.3.1]: https://github.com/promised-ai/lace/compare/python-0.3.0...python-0.3.1
[rust-0.3.1]: https://github.com/promised-ai/lace/compare/rust-0.3.0...rust-0.3.1
[python-0.3.0]: https://github.com/promised-ai/lace/compare/python-0.2.0...python-0.3.0
[rust-0.3.0]: https://github.com/promised-ai/lace/compare/rust-0.2.0...rust-0.3.0
[python-0.2.0]: https://github.com/promised-ai/lace/compare/python-0.1.2...python-0.2.0
[rust-0.2.0]: https://github.com/promised-ai/lace/compare/rust-0.1.2...rust-0.2.0
[python-0.1.2]: https://github.com/promised-ai/lace/compare/python-0.1.1...python-0.1.2
[python-0.1.1]: https://github.com/promised-ai/lace/compare/python-0.1.0...python-0.1.1
[rust-0.1.2]: https://github.com/promised-ai/lace/compare/rust-0.1.1...rust-0.1.2
[rust-0.1.1]: https://github.com/promised-ai/lace/compare/rust-0.1.0...rust-0.1.1
[python-0.1.0]: https://github.com/promised-ai/lace/releases/tag/python-0.1.0
[rust-0.1.0]: https://github.com/promised-ai/lace/releases/tag/rust-0.1.0

