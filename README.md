# Braid

Fast, transparent genomic analysis.

## Install

```bash
$ cargo build --release
```

### Flags

The build recognized a number of environment variables as flags.

#### `BRAID_NO_PAR_MASSFLIP` - disable massflip parallelism

Massflip is a large portion of the `finite_cpu` and `slice` algorithms.
Parallelism doesn't become much of a benefit until there are about 50k cells in
the massflip table. If parallelism is enabled, the build script will run a
number of benchmarks are determine the row and column threshold at which
parallelism should be used. For an $N \times K$ table parallelism will be used when

<center>$$ \epsilon \gt N^a N^b + c,$$</center>

where $\epsilon$ is the desired speedup ratio.

## Standard workflow

Start and oracle from a cleaned csv data file:

```bash
$ braid codebook myfile.csv myfile.codebook.yaml
$ braid run --csv myfile.csv --nstates 16 --n-iters 500 --codebook myfile.codebook.yaml myfile.braid 
$ braid oracle myfile.braid
```

## NOTEs
- Runtime comparisons w/ DL and RF aren't fair because RF and DL can only
  answer one question. Braid can answer tons of questions. One would have to
  multiply the RF/DL runtime by the number of questions to answer. Also, the
  more we run RF/DL, the more likely we are to be doing something we shouldn't,
  e.g. overfitting; trend fishing.

## Questions we'd like to answer
- [X] Which regions on the genome affect traits? (QTL)
    - Solved using `mi` or `depprob`
- [X] How does a SNP affect a trait
    - solved with `logp given`
- [X] Probability I will see trait `x = y` from a certain genotype
    - Solved with integral over `logp given`
- [X] What is the optimal genotype for a trait
    - Could be solved by enumerating the geneotypes in the QTL and computing
      the mean of `trait | genotype`.
    - Probably should let the user define their own optimization target
    - Could use greedy search
    - Search by all dependent regions or by specific QTLs

## Genomics-specific metadata
- [X] Each column needs to be identified as Genetic, Phenotypic, Environmental,
  or Other
  - This will help us with auto-analysis information
  - This can be a metadata additional to the codebook, no integrated with it,
    because braid has uses outside genomics
- [X] The server should be able to return the list of columns that fall under
  each type

## Sales materials (Demos)
- [ ] Pretty plots!
    - Graphs
- [ ] Phenotype data (nam282)
    + Show predictive capability via PPC
    + Compare against linear regression (as industry standard), Random Forests
      (RF), and Deep Learning (DL). Show QTL-like results using feature
      importance or whatever, then try to ask a different question
- [ ] Gene expression network?
    + Need graph viz in pybraid
- [ ] Precision medicine
- [ ] GxE
    + Synthetic data

## Future

### Usability and stability

- [X] Command line utility for auto-generating codebooks for CSVs.
- [X] `Savedata` object that stores data and dataless-states separately. We
  don't want to store a copy of the data for each state!
    + Probably going to be a directory (or archive) of split-up files
- [X] pairwise function like `mi`, `rowsim`, and `depprob` should be called
  with a vector of pairs, `Vec<(usize, usize)>` to reduce `Sever` RPC calls.
- [X] optional `comments` field in codebook
- [X] draw(row, col, n) method for oracle that simulates from a place in the
  table
- [X] Fix alpha prior so tables don't get crazy complex and take forever to run
- [X] View parallelism
    - [X] Row reassignment should be run in parallel
    - [X] Feature parameter reassignment
- [X] Benchmarks (log likelihood by time for different algorithms)
- [X] Gibbs Cols
- [X] Use RNG properly in parallel code
- [X] Easy cli command to launch example `Oracle`s
    - [X] Can do in pybraid
- [ ] incremental output in case runs are terminated early
- [X] Engine saves states once they're finished
    - [ ] Would be nice if we could send a kill signal, which would cause a
      stop and save after the current iteration is complete
- [ ] Logger messages from `engine.run`
- [X] Better: Allow alpha priors to be set from the codebook
- [X] Clean up some of the `impl Rng` by moving into type parameters, e.g.
      `fn func<R: Rng>(r: &mut R)`
- [ ] Split-merge rows
- [ ] GPU parallelism for row reassign
- [X] Gibbs Rows
    - [X] Current implementation is crazy inefficient because it doesn't use the
      suffstats properly
    - [X] Use suffstats properly
- [X] optimize discrete-discrete mutual information
- [X] "Improved slice algorithm" for columns and rows

### Development
- [X] Organize sourcefiles in idiomatic way
- [X] Continuous integration
- [X] Benchmarks
- [X] No compilations warning (use `Result<_>`)
- [X] Inference tests
- [X] Statistical tests in braid
    - [X] Ks test
    - [X] Chi-square test
    - [X] Gaussian kernel permutation test
- [X] Automatic regression testing
    - [X] Testing framework
    - [X] Reporting and storage web app
    - [X] Assets on aws, reporting live on web

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
