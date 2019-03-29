extern crate braid_flippers;
extern crate rand;

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use braid_flippers::{massflip_par, massflip_ser};
use rand::FromEntropy;

const N_BENCH_REPS: usize = 2;

fn run_bench(n: usize, k: usize) -> (u32, u32) {
    let mut rng = rand::XorShiftRng::from_entropy();

    let t_ser = (0..N_BENCH_REPS).fold(0_u32, |acc, _| {
        let xs: Vec<Vec<f64>> = vec![vec![0.0; k]; n];
        let start = Instant::now();
        massflip_ser(xs, &mut rng);
        let elapsed = start.elapsed();
        acc + elapsed.subsec_nanos()
    });

    let t_par = (0..N_BENCH_REPS).fold(0_u32, |acc, _| {
        let xs: Vec<Vec<f64>> = vec![vec![0.0; k]; n];
        let start = Instant::now();
        massflip_par(xs, &mut rng);
        let elapsed = start.elapsed();
        acc + elapsed.subsec_nanos()
    });

    (t_ser, t_par)
}

// Really crummy ordinary least squares regression for this one 2D data case
fn ols2(ns: Vec<usize>, ks: Vec<usize>, speedup: Vec<f64>) -> (f64, f64, f64) {
    let nf = speedup.len() as f64;
    let x: Vec<Vec<f64>> = ns
        .iter()
        .zip(ks.iter())
        .map(|(&n, &k)| vec![(n as f64).ln(), (k as f64).ln()])
        .collect();

    // compute offsets to center data for the OLS
    let (xn_offset, xk_offset) = x.iter().fold((0.0, 0.0), |acc, xi| {
        (acc.0 + xi[0] / nf, acc.1 + xi[1] / nf)
    });

    let y_offset = speedup.iter().fold(0.0, |acc, &y| acc + y) / nf;

    // compute (X^T X)^{-1}
    let xtx_inv: Vec<Vec<f64>> = {
        let mut xtx: Vec<Vec<f64>> = vec![vec![0.0; 2]; 2];
        x.iter().for_each(|xi| {
            xtx[0][0] += (xi[0] - xn_offset) * (xi[0] - xn_offset);
            xtx[0][1] += (xi[0] - xn_offset) * (xi[1] - xk_offset);
            xtx[1][0] += (xi[1] - xk_offset) * (xi[0] - xn_offset);
            xtx[1][1] += (xi[1] - xk_offset) * (xi[1] - xk_offset);
        });
        let xtx = xtx; // turn off mutability
        let det = (xtx[0][0] * xtx[1][1] - xtx[0][1] * xtx[1][0]).recip();
        vec![
            vec![xtx[1][1] * det, -xtx[0][1] * det],
            vec![-xtx[1][0] * det, xtx[0][0] * det],
        ]
    };

    // compute X^T Y
    let mut xty: Vec<f64> = vec![0.0; 2];
    x.iter().zip(speedup.iter()).for_each(|(xi, yi)| {
        xty[0] += (xi[0] - xn_offset) * (yi - y_offset);
        xty[1] += (xi[1] - xk_offset) * (yi - y_offset);
    });
    let xty = xty;

    // compute coeffs
    let bn = xty[0] * xtx_inv[0][0] + xty[1] * xtx_inv[0][1];
    let bk = xty[0] * xtx_inv[1][0] + xty[1] * xtx_inv[1][1];

    // choose parallelism when speedup greater than `t` times
    //          t < a*log(n) + b*log(k) + c
    //     t - c  < log(n^a) + log(k^b)
    //     t - c  < log(n^a * k^b)
    // exp(y - c) < n^a * k^b
    let intercept = y_offset - (xn_offset * bn + xk_offset * bk);

    (bn, bk, intercept)
}

fn main() {
    let nopar = cfg!(debug_assertions) || env::var("BRAID_NOPAR_ALL").is_ok();
    let nopar_col_assign = env::var("BRAID_NOPAR_COL_ASSIGN").is_ok() || nopar;
    let nopar_row_assign = env::var("BRAID_NOPAR_ROW_ASSIGN").is_ok() || nopar;
    let nopar_massflip = env::var("BRAID_NOPAR_MASSFLIP").is_ok() || nopar;

    let par_switches = format!(
        "\
        const NOPAR_COL_ASSIGN: bool = {};
        const NOPAR_ROW_ASSIGN: bool = {};
        ",
        nopar_col_assign, nopar_row_assign,
    );

    let mfs_fn = if nopar_massflip {
        String::from(
            "\
            #[inline]
            pub fn mfs_use_par(_k: usize, _n: usize) -> bool {{
                false
            }}
        ",
        )
    } else {
        let ks_in: Vec<usize> = vec![2, 5, 10, 20, 50, 75, 100, 175, 250];
        let ns_in: Vec<usize> = vec![
            10, 50, 100, 250, 500, 1_000, 5_000, 10_000, 25_000, 50_000,
            100_000,
        ];

        let mut ks: Vec<usize> = Vec::new();
        let mut ns: Vec<usize> = Vec::new();
        let mut speedup: Vec<f64> = Vec::new();
        for k in ks_in.iter() {
            for n in ns_in.iter() {
                let (t_ser, t_par) = run_bench(*n, *k);
                ks.push(*k);
                ns.push(*n);
                speedup.push(t_ser as f64 / t_par as f64);
            }
        }

        let (a, b, int) = ols2(ns, ks, speedup);
        format!("\
            const MASSFLIP_SWITCH_INTERCEPT: f64 = {};

            #[inline]
            pub fn mfs_use_par(k: usize, n: usize) -> bool {{
                let nf = n as f64;
                let kf = k as f64;
                nf.powf({}) * kf.powf({}) < (1.5 - MASSFLIP_SWITCH_INTERCEPT).exp()
            }}
        ", int, a, b)
    };

    let out_dir = env::var("OUT_DIR").unwrap();

    {
        let mfs_dest_path = Path::new(&out_dir).join("msf_par_switch.rs");
        let mut f_mfs = File::create(&mfs_dest_path).unwrap();
        f_mfs.write_all(mfs_fn.as_bytes()).unwrap();
    }

    {
        let switch_dest_path = Path::new(&out_dir).join("par_switch.rs");
        let mut f_switch = File::create(&switch_dest_path).unwrap();
        f_switch.write_all(par_switches.as_bytes()).unwrap();
    }
}
