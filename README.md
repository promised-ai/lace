# Braid

Humanistic AI backend.

[![pipeline status](https://gitlab.com/Redpoll/braid/badges/v0.26.1/pipeline.svg)](https://gitlab.com/Redpoll/braid/-/commits/v0.26.1)
[![coverage report](https://gitlab.com/Redpoll/braid/badges/v0.26.1/coverage.svg)](https://gitlab.com/Redpoll/braid/-/commits/v0.26.1)

## Install

### Install from Redpoll-Crates Repository

To install braid pre-built from the `redpoll-crates` Cloudsmith repository. Run the following:

```bash
cargo install braid --registry redpoll-crates
```

If you do not have this registry configured on your system, you will need
to follow the instructions
[here on Cloudsmith.io](https://cloudsmith.io/~redpoll/repos/crates/setup/#formats-cargo)
in order to do so. If this link does not work, you will need to have your
user added to the Cloudsmith repository.

### Build from Source

To build braid and its documentation from source, you may do the following:

In the root directory, build `braid`

```bash
$ cargo build --release
```

Build documentation

```bash
$ cargo doc --all --no-deps
```

Run tests

```bash
$ cargo test --all
```

Install binary to system

```bash
$ cargo install --path .
```

Note that when the build script runs, example files are moved to your data
directory.  Once you ask for an `Oracle` for one of the examples, braid will
build the metadata if it does not exist already. If you need to regenerate
the metadata — say the metadata spec has changed — you can do so with the
following CLI command:

```bash
$ braid regen-examples
```

## Standard workflow

Run inference on a csv file using the default codebook and settings, and save
to `mydata.braid`

```
$ braid run --csv mydata.csv mydata.braid
```

> _Note_: The CSV must meet a minimum set of formatting criteria:
> * The first row of the CSV must be a header
> * The first column of the csv must be "ID"
> * All columns in the csv, other than ID, must be in the codebook, if a codebook is specified
> * Missing data are empty cells
>
> For more on CSV formatting, see the cargo docs under the `braid::data::csv` module.

You can specify which transitions and which algorithms to use two ways. You can use CLI args

```
$ braid run \
    --csv mydata \
    --row-alg slice \
    --col-alg gibbs \
    --transitions=row_assignment,view_alphas,column_assignment,state_alpha \
    mydata.braid
```

Or you can provide a run config

```yaml
# runconfig.yaml
n_iters: 4
timeout: 60
save_path: ~
transitions:
  - row_assignment: slice
  - view_alphas
  - column_assignment: gibbs
  - state_alpha
  - feature_priors
```

```
$ braid run \
    --csv mydata \
    --run-config runconfig.yaml \
    mydata.braid
```

Note that any CLI arguments covered in the run config cannot be used if a run
config is provided.

## Future

### Prioritized TODOs
- [X] Doctests with test dataset loaders
- [ ] States should have runners that monitor the number of iterations and the
    estimated time remaining, and should be able to stop states early and
    write outputs
    - [ ] Run-progress monitoring via daemon process or similar. Should be
        able to get output via CLI.
    - [ ] incremental output in case runs are terminated early (`write_cpt` arg)
- [X] All mi uses quadrature, fallback to MC integration.

### Usability and stability
- [ ] PIT should work for discrete & categorical distributions
- [ ] Logger messages from `engine.run`
- [ ] incremental output in case runs are terminated early (`checkpoint` arg)
- [ ] Run-progress monitoring
- [ ] Use quadrature for MI when possible
    + Not really gonna be possible if using multivariate features
- [ ] Split-merge rows
- [ ] GPU parallelism for row reassign (meh)

### Development
- [ ] Work on intuitive naming and organization
- [X] `Given` type should be an enum
- [X] Broken categorical / discrete PIT
    - [X] Rename PIT to SampleError trait
    - [X] Implement `SampleError` as `Pit` for continuous distributions
    - [X] Implement `SampleError` as CDF error for Categorical?
    - [X] Tests!
- [X] Fix error kind names (should end with `Error`)
- [X] No `bool` switches
- [X] Better names for types like `FType` and `DType`
- [ ] Make as much as possible private
- [ ] More docs and doctests!

### Comparissons / Tests
- [ ] Vs industry standard QTL
    - [X] ANOVA
    - [ ] Interval Mapping
    - [ ] Empirical mutual information
- [ ] Vs predictive models (one pheno at a time & all at once)
    - [ ] Deep learning
    - [ ] Random Forest

### Scaling strategies

- Split-merge row
    + Pros: faster theoretical convergence
    + Cons: slower per iteration; serial
- Split-merge columns
    + Pros: faster theoretical convergence
    + Cons: Serial, MCMC algorithm doesn't exist yet. Will be complicated.
- Gpu Parallelism
    + Pros: faster for large data
    + Cons: data size limited by GPU mem size
    + Notes:
        - The approach would be to store a copy of the data on the gpu then
          have function to basically to what the finite cpu algorithm does, but
          do it all on the gpu. Then all we'd need to do is to pull an `nrows`
          length vector of vector or double off the Gpu. Or maybe we could do
          the `massflip` operation on the gpu as well, then simply pull off a
          vector of `usize`.
        - We can't do the massflip on the GPU if the data is spread on multiple
          GPUs.
        - The computation for computing likelihoods is very quick, so there has
          to be a lot of data for this to overwhelm the i/o.
- Multi-machine engine parallelism (one state per machine)
    + Pros: Fast, simple. Works regardless of algorithm.
    + Cons: Requires infrastructure.
- Multi-machine state parallelism (one view per machine)
    + Pros: Maximal parallelism
    + Cons: Difficult to engineer; lot of i/o overhead
    + Notes:
        - Let's say that each machine gets a view, we can reduce io overhead by
          doing multple sweeps per iterion. When the assignment of columns to
          views is updated the data views are reformed in the cluster (one view
          per machine). The row assignment if update by several sweeps before
          the column reassignment happens.
