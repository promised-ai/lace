# Braid

Fast, transparent genomic analysis.

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

- [ ] `Savedata` object that stores data and dataless-states separately. We
  don't want to store a copy of the data for each state!
- [ ] pairwise function like `mi`, `rowsim`, and `depprob` should be called
  with a vector of pairs, `Vec<(usize, usize)>` to reduce `Sever` RPC calls.
- [X] optional `comments` field in codebook
- [ ] incremental output in case runs are terminated early

### Development
- [ ] Continuous intergration
- [ ] Benchmarks
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
