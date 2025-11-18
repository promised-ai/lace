# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [rust-0.9.0] - 2025-11-19

### Changed

- *Prior version metadata is incompatible with this version*
- Updated `rv` and `rand` dependencies
- Removed all subcrates and collapsed into single `lace` crate
- Categorical variables can take on more than 256 distinct values because
  categorical data are now represented with `u32` instead of `u8`.

## [python-0.9.0] - 2025-11-19

### Changed
- Prior version metadata is incompatible with this version
- Updated to lace 0.9.0
- Categorical variables can take on more than 256 distinct values

## [rust-0.8.0] - 2024-04-10

### Added

- Added ability Pitman-Yor prior process
    - Users now specify the prior process (and its prior) in the codebook
- `lace_stats::prior_process::Builder`

### Changed

- Offloaded a lot of `Assignment` functionality to `PriorProcess`
- `StateAlpha` and `ViewAlpha` transitions are now `StatePriorProcessParams` and `ViewPriorProcessParams`
- Changed `SerializedType` default to `Bincode`
- moved `Assignment` from `lace_cc` to `lace_stats`

## [python-0.8.0] - 2024-04-10

### Added

- Added ability Pitman-Yor prior process.
- Added `remove_rows` to `Engine`.
- Added `with_index` to `CodeBook`.

### Changed

- `StateAlpha` and `ViewAlpha` transitions are now `StatePriorProcessParams` and `ViewPriorProcessParams`
- Updated Pyo3 version to 0.21

## [python-0.7.1] - 2024-02-27

### Added

- Added support for pickling `lace.Engine`
- Improved error reports for certain methods

### Changed

- Updated dependencies for pylace:
    `pandas`: 1.3 -> 2.2
    `pyarrow`: 14.0 -> 15.0

### Fixed

- Fixed `__version__` property

## [python-0.7.0] - 2024-02-07

### Added

- `Engine.update` uses `tqdm.auto` for progress bar reporting, to support progress bars in notebooks
- Added `flat_columns` option to `Engine` constructor to enable creating engines with one view
- Added `variability` method to `Engine`

### Changed

- Improved errors in type conversions.
- Rearranged test execution

### Fixed

- Default transition sets now hit all required transitions
- Fixed typo in internal `Dimension` class

## [rust-0.7.0] - 2024-02-07

### Added

- `DataParseError::CodebookAndDataRowsMismatch` variant for when the number of rows in the codebook and the number of rows in the data do not match.
- `DataParseError::DataFrameMissingColumn` variant for when a column is in the codebook but not in the initial dataframe.
- Added `variability` method to `OraceT` trait

### Changed

- Added parallelism to `Slice` row reassignment kernel. Run time is ~6x faster.

### Fixed

- Initializing an engine with a codebook that has a different number of rows than the data will result in an error instead of printing a bunch of nonsense.

## [python-0.6.0] - 2024-01-23

### Added

- Added support for Python 3.12
- Added `plot.state` function to render PCC states
- Added `analysis.explain_prediction` to explain predictions
- Added `plot.prediction_explanation` to render prediction explanations
- Added `analysis.held_out_uncertainty`
- Added `analysis.attributable_[neglogp | inconsistrncy | uncertainty]` to quantify the amount of surprisal (neglogp), inconsistency, and uncertainty attributable to other features
- Added `utils.predict_xs`

### Changed

- Updated all packages to have the correct SPDX for the Business Source License
- Changed to using total variation distance for uncertainty prediction (see docs)
- Updated dependencies:
  - `numpy`: 1.21 -> 1.26
  - `polars`: 0.16.14 -> 0.20.5
  - `scipy`: 1.7 -> 1.11
  - `plotly`: 5.14 -> 5.18
  - `pyarrow`: 11 -> 14
- Added new dependencies: `seaborn`, `matplotlib`

### Fixed

- Fixed issue that would cause random row order when indexing pylace Engines by a single (column) index, e.g., engine['column'] would return the columns in a different order every time the engine was loaded
- Fixed bug in appending data with a boolean column

## [rust-0.6.0] - 2024-01-23

### Changed

- Updated all packages to have the correct SPDX for the Business Source License
- Removed internal implimentation of `logsumexp` in favor of `rv::misc::logsumexp`
- Changed to using total variation distance for uncertainty prediction (see docs)
- Bump min rust version to `1.62` to support `f64::total_cmp`
- Update to rv 0.16.2
- Update to Polars 0.36

### Fixed

- Fixed bug in appending data with a boolean column

## [python-0.5.0] - 2023-11-20

### Added

- Codebook.value_map fn and iterator

### Changed

- Added common transitions sets by name to `Engine.update` transition argument.

## [rust-0.5.0] - 2023-11-20

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
- Initial release on [PyPi](https://pypi.org/)

## [rust-0.1.0] - 2023-04-20

### Added
- Initial release on [crates.io](https://crates.io/)

[unreleased]: https://github.com/promised-ai/lace/compare/python-0.8.0...HEAD
[rust-0.9.0]: https://github.com/promised-ai/lace/compare/rust-0.8.0...rust-0.9.0
[python-0.9.0]: https://github.com/promised-ai/lace/compare/python-0.8.0...python-0.9.0
[rust-0.8.0]: https://github.com/promised-ai/lace/compare/rust-0.7.0...rust-0.8.0
[python-0.8.0]: https://github.com/promised-ai/lace/compare/python-0.7.1...python-0.8.0
[python-0.7.1]: https://github.com/promised-ai/lace/compare/python-0.7.0...python-0.7.1
[python-0.7.0]: https://github.com/promised-ai/lace/compare/python-0.6.0...python-0.7.0
[rust-0.7.0]: https://github.com/promised-ai/lace/compare/rust-0.6.0...rust-0.7.0
[python-0.6.0]: https://github.com/promised-ai/lace/compare/python-0.5.0...python-0.6.0
[rust-0.6.0]: https://github.com/promised-ai/lace/compare/rust-0.5.0...rust-0.6.0
[python-0.5.0]: https://github.com/promised-ai/lace/compare/python-0.4.1...python-0.5.0
[rust-0.5.0]: https://github.com/promised-ai/lace/compare/rust-0.4.1...rust-0.5.0
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

