#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate itertools;
extern crate rand;
extern crate serde_json;

mod bench;
mod ppc;
mod shapes;

use std::collections::BTreeMap;

use self::rand::prng::XorShiftRng;
use self::rand::SeedableRng;

use bench::{run_benches, BenchmarkResult};
use ppc::{run_ppc, PpcDataset, PpcDistance};
use shapes::{run_shapes_tests, ShapeResult};

#[derive(Serialize)]
struct RegressionResult {
    shapes: Vec<ShapeResult>,
    benchmark: Vec<BenchmarkResult>,
    ppc: BTreeMap<String, Vec<PpcDistance>>,
}

pub fn main() {
    let n_ks = 1000;
    let n_perm = 500;
    let n_perms = 1000;
    let seed: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let mut rng = XorShiftRng::from_seed(seed);

    let ppc_res = run_ppc(vec![PpcDataset::Animals(8, 500)], 1000, &mut rng);
    let shapes_res = run_shapes_tests(n_ks, n_perm, n_perms, &mut rng);
    let bench_res = run_benches(&mut rng);

    let result = RegressionResult {
        shapes: shapes_res,
        benchmark: bench_res,
        ppc: ppc_res,
    };

    let res_str = serde_json::to_string(&result).unwrap();
    println!("{}", res_str)
}
