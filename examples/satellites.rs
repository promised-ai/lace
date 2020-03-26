// Compute the mutual information and dependence probability for every pair of
// variables in the satellites dataset.
use braid::examples::satellites::Column;
use braid::examples::Example;
use braid::{MiType, OracleT};
use rayon::prelude::*;
use std::convert::TryInto;

fn main() {
    // Load the satellites example
    let oracle = Example::Satellites.oracle().unwrap();

    let mut col_pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..20 {
        for j in 0..20 {
            col_pairs.push((i, j));
        }
    }

    println!("{:>51} │ {:>10.5} │ {:>10.5} ", "Columns", "MI", "DepProb");
    println!(" {:─>52}{:─>13}{:─>12}", "┼", "┼", "");

    col_pairs.par_iter().for_each(|(col_a, col_b)| {
        // Use samples to approximate mutual information between pairs of
        // continuous variables. Negative mutual information indicates an
        // approximation error which can often be fixed with more samples
        // (longer compute time).
        let mi = oracle
            .mi(*col_a, *col_b, 1_000_000, MiType::UnNormed)
            .unwrap();

        let depprob = oracle.depprob(*col_a, *col_b).unwrap();

        let col_name_a: Column = (*col_a).try_into().unwrap();
        let col_name_b: Column = (*col_b).try_into().unwrap();
        let label = format!("{:?}, {:?}", col_name_a, col_name_b);

        println!("{:>51} │ {:>10.5} │ {:10.5}", label, mi, depprob)
    });
}
