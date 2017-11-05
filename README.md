# Braid

Fast, transparent genomic analysis.

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