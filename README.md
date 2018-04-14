# Braid

Fast, transparent genomic analysis.

## Standard workflow

Start and oracle from a cleaned csv data file:

```bash
$ braid codebook myfile.csv myfile.codebook.yaml
$ braid run --csv myfile.csv 16 500 myfile.braid myfile.codebook.yaml
$ braid oracle myfile.braid
```

## Questions we'd like to answer
- [X] Which regions on the genome affect traits? (QTL)
    - Solved using `mi` or `depprob`
- [X] How does a SNP affect a trait
    - solved with `logp given`
- [X] Probability I will see trait `x = y` from a certain genotype
    - Solved with integral over `logp given`
- [ ] What is the optimal genotype for a trait
    - Could be solved by enumerating the geneotypes in the QTL and computing
      the mean of `trait | genotype`.
    - Probably should let the user define their own optimization target
    - Could use greedy search
    - Search by all dependent regions or by specific QTLs

## Genomics-specific metadata
- [ ] Each column needs to be identified as Genetic, Phenotypic, Environmental,
  or Other
  - This will help us with auto-analysis information
  - This can be a metadata additional to the codebook, no integrated with it,
    because braid has uses outside genomics
- [ ] The server should be able to return the list of columns that fall under
  each type

## Future

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
- Multi-machine engine parallesim (one state per machine)
    + Pros: Fast, simple. Works regardless of algorithm.
    + Cons: Requires infrastructure.
- Multi-machine state parallesim (one view per machine)
    + Pros: Maximal paralleism
    + Cons: Difficult to engineer; lot of i/o overhead
    + Notes:
        - Let's say that each machine gets a view, we can reduce io overhead by
          doing multple sweeps per iterion. When the assignment of columns to
          views is updated the data views are reformed in the cluster (one view
          per machine). The row assignment if update by several sweeps before
          the column reassignment happens.

### Usability and stability

- [X] Command line utility for auto-generating codebooks for CSVs.
- [X] `Savedata` object that stores data and dataless-states separately. We
  don't want to store a copy of the data for each state!
    + Probably going to be a direcory (or archive) of split-up files
- [X] pairwise function like `mi`, `rowsim`, and `depprob` should be called
  with a vector of pairs, `Vec<(usize, usize)>` to reduce `Sever` RPC calls.
- [X] optional `comments` field in codebook
- [X] draw(row, col, n) method for oracle that simulates from a place in the
  table
- [X] Fix alpha prior so tables don't get crazy complex and take forever to run
- [ ] Engine saves states once their finished
    - [ ] Would be nice if we could send a kill signal, which would cause a
      stop and save after the current iteration is complete
- [ ] Logger messages from `engine.run`
- [ ] Better: Allow alpha priors to be set from the codebook
- [ ] View parallelism
    - [X] Row reassignment should be run in parallel
    - [ ] Feature parameter reassignment
- [ ] Use RNG propoerly in parallel code
- [ ] incremental output in case runs are terminated early
- [ ] Easy cli command to launch example `Oracle`s
- [ ] optimize discrete-discrete mutual information

### Development
- [ ] Organize sourcefiles in idomatic way
- [X] Continuous intergration
- [X] Benchmarks
- [ ] No compilations warning (use `Result<_>`)
- [ ] Inference tests (can be done w/ `pybraid`)
- [ ] Statistical tests in braid
    - [X] Ks test
    - [X] Chi-square test
    - [ ] Gaussian kernel permutation test
- [ ] Automatic regression testing

## Random Variate Examples

```rust
extern crate braid;

use braid::dist::{Delta, Gaussian, Gamma};
use braid::rv;

fn main() {
    let prior =  rv::GaussianPrior{mu: Gaussian{0.0, 1.0}, sigma: Delta{1.0}};
    let rv = rv::Rv::new(prior);
}
```
