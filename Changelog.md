# Changelog

## 0.33.0
- Created `braid_metadata` subcrate to handle creating and converting metadata
- Add metadata encryption
- Moved data things into `braid_data`
- Moved cross cat components into `braid_cc` crate
- Refactor `Index` into `TableIndex`
- Changed `Row.row_name` field from `String` to `RowIndex`
- Changed `Value.col_name` field from `String` to `ColumnIndex`
- Added variants to `InsertDataError`
- Renamed methods in `TranslateDatum` trait
- Updated rv to v0.10
- Changed the way prediction is done on continuous columns to achieve a 4x
    speedup.
- Changed the way continuous-categorical mutual information is computed for a
    40x speedup
- Added a bunch of top-level convenience re-imports
- No timeout by default in `braid run` commands
- No default number of iterations in `braid run`. User must enter `-n` or
    `--n-iters` manually
- cli `codebook` function much faster
- cli `codebook` function no longer supports the experimenal `Labeler` type
- cli `codebook` function has `--no-check` argument to skip sanity checking
- Count type hyper prior is Gamma(a, 1) on shape and Gamma(b, 1) on rate
- Parallelized the Gibbs column reassignment kerneel for significant speedup
- Fixed bug in `Bencher` where column and row reassignment algorithms inside the
    update configuration transitions were ignored

## 0.32.3
- Add '--flat-columns' argument to `braid run` command so that an engine can be
    run with a flat view structure. The user will need to manually define the
    transitions not to run the column assignment transition if they wish to keep
    the flat structure during the run.
- Fixed bug where not all transitions could be passed to `braid run
    --transitions`

## 0.32.2
- Add `Engine.remove_data` method to remove rows, columns, and cells from an
    engine. Several other functions added in support.

## 0.32.1
- Fix bugs in sparse container that cause fragmenting
- Improve Gibbs column reassignment kernel by considering more singleton views
- Tighten Continuous geweke prior any hyper to make tests behave better
- Fix bug in gewke from_prior in view and state that caused geweke tests to
    fail. Note that the algorithms were never invalid, the test was invalid.

## 0.32.0
- Adjusting default Continuous hyper prior to avoid infinite sigma on posterior
    draw and decrease overfitting.
- Added an example to `Engine.insert_data`
- small optimization in `Engine.insert_data`

## 0.31.0
- To increase interpretability, the prior on `Continuous` columns is now normal
    inverse chi-squared. The hyper prior is changed as well.
- Priors parameters are re-sampled using different samplers
- Added examples of fitting to a ring (`examples/shapes.rs`)

## 0.30.0
- Add optional `prior` field to `Codebook` `ColTypes`. If `prior` is set while
    loading an `Engine` from a csv, the hyper prior will be ignored and the
    prior parameters will not be updated.
- Add `col_weighted` argument to `Oracle.rowsim` that weights row similarity by
    the number of columns instead of the number of views. Rows that have more
    columns cells in the same category will have higher similarity in this mode.
- Add `DatalessOracle` for data-sensitive applications. `DatalessOracle` does
    not store its data so it can only perform inference operations, including
    simulation for synthetic data generation.
- `Engine` and `Oracle` implement `TryFrom<Metadata>` instead of `From` and will
    fail to convert if the Metadata does not contain the data.
- Removed newtype priors. Priors and hypers are now defined separately in
    columns.

## 0.29.2
- Fix bug when deleting 0 rows from an `Engine`.

## 0.29.1
- Switch to buffered reader in file_utils to speed up reads

## 0.29.0
- Add caches to `Column` and `ConjugateComponent` to speed up computation of
    marginals and posterior predictives. Gibbs and Sams row kernels at ~32%
    faster for Categorical data.
- Add `del_rows_at` method to `Engine` to delete rows.

## 0.28.0
- Added `append_strategy` field to `Engine::insert_data`'s `WriteMode` argument.
    This allows users to treat the crosscat table like a sliding window over a
    stream of data.
- Added method to delete rows from the Engine

## 0.27.3
- Added `flatten_cols` method to `Engine` and `cc::State` that assigns all
    columns to a single view.

## 0.27.2
- Optimized update_prior
    + update_prior for Continuous ~41x faster (50 component models)
    + update_prior for Categorical ~32x faster (50 component models w/ 4
        categories)
    + update_prior for Count ~3x faster (50 component models)
- Fix bug where inserting new rows into an `Engine` can cause there to be more
    components than weights in a `View` unless an `Engine::update` or
    `Engine::run` is performed.

## 0.27.1
- braid_stats produces a better error message when the gaussian posterior is
    invalid.

## 0.27.0
- Do not allow users to insert non-finite data into the engine via
    `Engine.insert_data`. Added a new error variant.

## 0.26.3
- Added methods to `cc::View` and `cc::State` to run Gibbs reassignment
  transitions on a specific column and row.
- Added `Engine::cell_gibbs`, which runs Gibss on a single column and view-row.
- Implemented `Default` for `InsertDataActions`

## 0.26.2
- Fixed bug where users could `Engine::insert_data` of the wrong type into a new
  column

## 0.26.1
- Fixed bug in `StateBuilder` which resulted in degenerate sates. User-supplied
  nrows would be used even if the user passed features

## 0.26.0
- Catch an bug on `Engine` construction caused when non-empty `row_names` or
  `column_metadata` are supplied alongside an empty data source.
- `View`, `State`, `Engine`, and `Oracle` implement `Debug`

## 0.25.0
- Added ability to extend the support of individual `Engine` columns using
  `insert_data`. For example, inserting `2` into a binary column may -- if
  allowed by the user -- transform that column to a ternary column.

## 0.24.1
- Added ability to choose which example(s) to regenerate from the cli using the
  `--examples` flag. Defaults to animals and satellites.
- Added ability to specify `n_iters` and `timeout` for examples re-generation

## 0.24.0
- Implemented new data structure for columnar data in `braid_data` subcrate.
  FiniteCpu and Slice row assignment algorithms are now significantly faster,
  but Gibbs and Sams are slightly (~3%) slower due to the increased cost of
  indexing.

## 0.23.1
- Removed prettytable-rs dependency due to audit warning. `braid summarize`
  tables now render differently.

## 0.23.0
- Improve errors in `OracleT::logp`
- Fix bug where users could pass `Datum::Missing` to `OracleT::logp` targets
  and conditions.
- Fix bug that caused a error when directories named `<name>.state` were nested
  under the braidfile.
- Implement `Mode<Label>` for Labeler
- Upgrade to rv 0.10.0
- Added `OracleT::logp_scaled` method

## 0.22.0
- Remove SQLite

## 0.21.4
- Fix bug where running the Gibbs column transition on a state with a single
  column broke everything.

## 0.21.3
- Improve documentation of some OracleT functions

## 0.21.2
- Fix bug that causes a crash when attempting to call `update` or `run` on an
  empty `Engine`.

## 0.21.1
- Change the way empty sufficient statistics are created in the `Column` so
  that there do not have to be instantiated components.

## 0.21.0
- Row and column reassignment kernel name is passed as a field in the
  `RowAssignement` and `ColumnAssignment` transition
  + Changes the way CLI args work for run. User can now pass a run config to
    specify which transitions to run using which kernels

## 0.20.5
- Add sequentially adaptive merge-split (SAMS) row-reassignment algorithm
- Fix bug in braid_geweke that caused tests to break for gibbs and sams

## 0.20.4
- View accum score is parallel
- Fixed bug in Gaussian::accum_score_par that produced incorrect ln PDF

## 0.20.3
- Examples datasets timeout after 30 seconds

## 0.20.2
- Fix bug where user could insert a row with no values into an Engine.

## 0.20.1
- More robust continuous entropy calculation

## 0.20.0
- Added Count column type with Poisson likelihood
- Change the GewekeResult struct stores its internal data (breaks backward
  compatibility)
- Geweke AUCs compute MUCH faster O(n) -> O(log n)
- Internal improvements including heavy use of macros to reduce code repeated
  when adding a new column type
- No more Sobol QMC for joint entropy due to accuracy issues associates with
  joint distributions with long tails
- Internal improvements to simulation

## 0.19.0
- Errors revamped

## 0.18.7
- Miscellaneous internal improvements

## 0.18.6
- Properly save and log RNG state in braidfile
- Add `log_prior` state diagnostics

## 0.18.5
- Make `Metadata` public.

## 0.18.4
- Add min, max, and median number of categories in a view to State diagnostics

## 0.18.3
- Serialize and deserialize `Engine` and `Oracle` to `Metadata`

## 0.18.2
- Engine seed control works
- Fixed a bug where generating a `rv` `Mixture` distribution from a column
  would sometimes have zero-valued weights, which `rv` will not accept.

## 0.18.1
- Fix bug that caused continuous predictions to be wrong when there are
  multiple modes far apart.
