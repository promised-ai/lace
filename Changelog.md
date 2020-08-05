# Changelog

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
