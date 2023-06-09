# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [python-0.2.0] - Unreleased

### Changed

- `lace.Engine.__init__` now takes a Pandas' or Polars' dataframe to initialize an `Engine`.

### Added

- `Engine.load` supports loading metadata from disk.
- `lace.CodebookBuilder` supports loading a codebook from disk or using parameterized inference for creating a new `Engine`.

## [rust-0.2.0] - Unreleased

### Added

- `DataSource::Polars` to support direct loading of Polars' DataFrame.

### Fixed

- Clippy Lint in `lace-stats`.

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

[unreleased]: https://github.com/promised-ai/lace/compare/python-0.1.0...HEAD
[rust-0.1.1]: https://github.com/promised-ai/lace/compare/rust-0.1.1...rust-0.1.0
[python-0.1.0]: https://github.com/promised-ai/lace/releases/tag/python-0.1.0
[rust-0.1.0]: https://github.com/promised-ai/lace/releases/tag/rust-0.1.0

