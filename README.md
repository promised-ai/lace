# Braid

Fast, transparent genomic analysis.

## Install

```bash
$ cargo build --release
```

Build documentation

```bash
$ cargo docs --all --no-deps
```

Runing tests

```bash
$ bash scripts/test-setup.sh
$ cargo test --all
```

## Standard workflow

Run inference on a csv file using the default codebook and settings, and save
to `mydata.braid`

```
$ braid run --csv mydata.csv mydata.braid
```

### Flags

The build recognized a number of environment variables as flags.

#### `BRAID_NOPAR_ALL` - disable all parallelism

All parallelism is deactivated in debug mode.

#### `BRAID_NOPAR_MASSFLIP` - disable massflip parallelism

Massflip is a large portion of the `finite_cpu` and `slice` algorithms.
Parallelism doesn't become much of a benefit until there are about 50k cells in
the massflip table. If parallelism is enabled, the build script will run a
number of benchmarks are determine the row and column threshold at which
parallelism should be used. For an $N \times K$ table parallelism will be used when

```math
\epsilon \gt N^a N^b + c,
```

where $`\epsilon`$ is the desired speedup ratio.

#### `BRAID_NOPAR_COL_ASSIGN` - disable column assignment parallelism

The column scores are computed in parallel for each column for the `slice` and
`finite_cpu` columns.

#### `BRAID_NOPAR_ROW_ASSIGN` - disable column assignment parallelism

Since the row assignment of the columns in a view are independent of all other
columns' assignment, we can reassign the rows for each view in parallel.

## Future

### Prioritized TODOs
- [ ] Doctests with test dataset loaders
- [ ] States should have runners that monitor the number of iterations and the
    estimated time remaining, and should be able to stop states early and
    write outputs
    - [ ] Run-progress monitoring via daemon process or similar. Should be
        able to get output via CLI.
    - [ ] incremental output in case runs are terminated early (`write_cpt` arg)
- [ ] All mi uses quadrature, fallback to MC integration.

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
