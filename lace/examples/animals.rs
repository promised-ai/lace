// Determine the most predictable variables by using a predictor search for
// the set of N most predictive variables
use lace::examples::animals::Column;
use lace::examples::Example;
use lace::OracleT;
use std::convert::TryInto;
use std::io;
use std::io::prelude::*;

// Get this many predictors
const N: usize = 8;

fn main() {
    // Load the animals example
    let oracle = Example::Animals.oracle().unwrap();

    // display the information proportion, the target, and the set of predictors
    println!("I(<info_prop>) <target> => [<predictor_1>, ..., <predictor_N>]");

    // Try for each of the 85 columns in the dataset
    let mut output: Vec<_> = (0..85)
        .map(|col_ix| {
            // Get the best set of N predictors for this column
            let predictors: Vec<(Column, f64)> = oracle
                .predictor_search(&[col_ix], N, 10_000)
                .unwrap()
                .drain(..)
                .map(|(ix, info_prop)| (ix.try_into().unwrap(), info_prop))
                .collect();

            // Printing output
            let col: Column = col_ix.try_into().unwrap();
            let pred_cols: Vec<_> =
                predictors.iter().map(|(col, _)| col).collect();

            let ip = predictors[N - 1].1;

            let out = format!("I({:.4}) {:?} => {:?}", ip, col, pred_cols);

            // Cheap progress bar
            print!(".");
            io::stdout().flush().expect("Could not flush stdout");

            (ip, out)
        })
        .collect();
    println!();

    // Sort columns in descending order by information proportion
    output.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    output.reverse();

    // Print output
    output.iter().for_each(|(_, out)| println!("{}", out));
}
