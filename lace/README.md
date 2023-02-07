# Lace

Humanistic AI backend.

[![pipeline status](https://gitlab.com/Redpoll/lace/badges/v0.26.1/pipeline.svg)](https://gitlab.com/Redpoll/lace/-/commits/v0.26.1)
[![coverage report](https://gitlab.com/Redpoll/lace/badges/v0.26.1/coverage.svg)](https://gitlab.com/Redpoll/lace/-/commits/v0.26.1)

## Install

### Install from Redpoll-Crates Repository

To install lace pre-built from the `redpoll-crates` Cloudsmith repository. Run the following:

```bash
cargo install lace --registry redpoll-crates
```

If you do not have this registry configured on your system, you will need
to follow the instructions
[here on Cloudsmith.io](https://cloudsmith.io/~redpoll/repos/crates/setup/#formats-cargo)
in order to do so. If this link does not work, you will need to have your
user added to the Cloudsmith repository.

### Build from Source

To build lace and its documentation from source, you may do the following:

In the root directory, build `lace`

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
directory.  Once you ask for an `Oracle` for one of the examples, lace will
build the metadata if it does not exist already. If you need to regenerate
the metadata — say the metadata spec has changed — you can do so with the
following CLI command:

```bash
$ lace regen-examples
```

### Locking lace to a specific machine

To ensure that lace is only run on a specific machine, you may generate a hardware ID.

```console
$ cargo install rp-machine-id --registry redpoll-crates --features cli
$ rp-machine-id
UlAB8wQc6srvhu98uGsRflTZUrnQpseDrkp_9zN91482HYE
```

To lock the binary, pass the ID via the `BRAID_MACHINE_ID` env arg to the
`idlock` feature during compilation

```console
$ BRAID_MACHINE_ID=UlAB8wQc6srvhu98uGsRflTZUrnQpseDrkp_9zN91482HYp cargo build --features idlock
```

Now that binary will only work on the machine with the above ID.

You can also add a expiraiton date with the `BRAID_LOCK_DATE` env arg. The date
must be YYYY-MM-DD format, or you will cause runtime error.

```console
$ export BRAID_MACHINE_ID=UlAB8wQc6srvhu98uGsRflTZUrnQpseDrkp_9zN91482HYp 
$ export BRAID_LOCK_DATE=2023-03-15
$ cargo build --features idlock
```

 **Warning**: If you are building for a customer, you will want to disable the
 `dev` feature by passing the `--no-default-features` flag to cargo, which will
 remove the lace bench, regression, and regen-examples commands

```console
$ export BRAID_MACHINE_ID=UlAB8wQc6srvhu98uGsRflTZUrnQpseDrkp_9zN91482HYp 
$ export BRAID_LOCK_DATE=2023-03-15
$ cargo build --no-default-features --features idlock
```

## Standard workflow

Run inference on a csv file using the default codebook and settings, and save
to `mydata.lace`

```
$ lace run --csv mydata.csv mydata.lace
```

> _Note_: The CSV must meet a minimum set of formatting criteria:
> * The first row of the CSV must be a header
> * The first column of the csv must be "ID"
> * All columns in the csv, other than ID, must be in the codebook, if a codebook is specified
> * Missing data are empty cells
>
> For more on CSV formatting, see the cargo docs under the `lace::data::csv` module.

You can specify which transitions and which algorithms to use two ways. You can use CLI args

```
$ lace run \
    --csv mydata \
    --row-alg slice \
    --col-alg gibbs \
    --transitions=row_assignment,view_alphas,column_assignment,state_alpha \
    mydata.lace
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
$ lace run \
    --csv mydata \
    --run-config runconfig.yaml \
    mydata.lace
```

Note that any CLI arguments covered in the run config cannot be used if a run
config is provided.

## Encrypted metadata

Lace metadata can be encrypted using shared key encryption. Keys are 256-bits,
which is 64 hex characters. To generate a key:

```console
$ lace keygen
dccd7857a52e609b4ff95469fbe3478984021abbf800516ed847d59983f6221b
```

To run and save encrypted metadata simply pass the key in through the
`-k/--encryption-key` argument.

```console
$ lace run --csv data.csv -k $MY_KEY output.lace
```

Similarly, to add iterations to an encrypted engine.

```console
$ lace run --engine output.lace -k $MY_KEY output.lace
```

## Future

### Prioritized TODOs
- [ ] States should have runners that monitor the number of iterations and the
    estimated time remaining, and should be able to stop states early and
    write outputs
    - [ ] Run-progress monitoring via daemon process or similar. Should be
        able to get output via CLI.
    - [ ] incremental output in case runs are terminated early (`write_cpt` arg)

### Usability and stability
- [ ] PIT should work for discrete & categorical distributions
- [ ] Logger messages from `engine.run`
- [ ] incremental output in case runs are terminated early (`checkpoint` arg)
- [ ] Run-progress monitoring

### Development
- [ ] Make as much as possible private
- [ ] More docs and doctests!

### Scaling strategies
- Smart-dumb, dumb-smart split-merge
    + Pros: Makes better use of moves that naive split-merge
    + Cons: Complicated to implement
- Split-merge columns
    + Pros: faster theoretical convergence
    + Cons: Serial, MCMC algorithm doesn't exist yet. Will be complicated.
