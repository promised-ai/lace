extern crate rand;

use std::ops::AddAssign;
use self::rand::Rng;
use std::f64::INFINITY;
use std::f64::NEG_INFINITY;


pub fn minf64(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        panic!("Empty container");
    }
    xs.iter().fold(INFINITY, |m, &x| if x < m {x} else {m})
}

pub fn cumsum<T>(xs: &[T]) -> Vec<T>
    where T: AddAssign + Clone
{
    let mut summed: Vec<T> = xs.to_vec();
    for i in 1..xs.len() {
        summed[i] += summed[i - 1].clone();
    }
    summed
}


pub fn argmax(xs: &[f64]) -> usize {
    if xs.is_empty() {
        panic!("Empty container");
    }
    let mut maxval = NEG_INFINITY;
    let mut max_ix: usize = 0;
    for (ix, &x) in xs.iter().enumerate() {
        if x > maxval {
            maxval = x;
            max_ix = ix;
        }
    }
    max_ix
}


pub fn logsumexp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        panic!("Empty container");
    } else if xs.len() == 1 {
        xs[0]
    } else {
        let min = minf64(xs);
        xs.iter().fold(0.0, |acc, x| acc + (x - min).exp()).ln() + min
    }
}


pub fn pflip(weights: &[f64], mut rng: &mut Rng) -> usize {
    if weights.is_empty() {
        panic!("Empty container");
    }
    let ws: Vec<f64> = cumsum(&weights);
    let scale: f64 = *ws.last().unwrap();
    let r = rng.next_f64() * scale;

    match ws.iter().position(|&w| w > r) {
        Some(ix) => ix,
        None     => {
            let wsvec = weights.to_vec();
            panic!("Could not draw from {:?}", wsvec)},
    }

}


pub fn log_pflip(log_weights: &[f64], mut rng: &mut Rng) -> usize {
    let minval = minf64(log_weights);
    let mut weights: Vec<f64> = log_weights.iter().map(|w| (w-minval).exp()).collect();

    // doing this instead of calling pflip shaves about 30% off the runtime.
    for i in 1..weights.len() {
        weights[i] += weights[i-1];
    }

    let scale = *weights.last().unwrap();
    let r = rng.next_f64() * scale;

    match weights.iter().position(|&w| w > r) {
        Some(ix) => ix,
        None     => {panic!("Could not draw from {:?}", weights)},
    }
}


pub fn massflip<R: Rng>(mut logps: Vec<Vec<f64>>, rng: &mut R) -> Vec<usize> {
    let k = logps[0].len();
    let mut ixs: Vec<usize> = Vec::with_capacity(logps.len());

    for lps in logps.iter_mut() {
        let minval = minf64(lps);
        lps[0] -= minval;
        lps[0] = lps[0].exp();
        for i in 1..k {
            lps[i] -= minval;
            lps[i] = lps[0].exp();
            lps[i] += lps[i-1]
        }

        let scale: f64 = *lps.last().unwrap();
        let r: f64 = rng.gen_range(0.0, scale);

        // Is a for loop faster?
        let ct: usize = lps.iter().fold(0, |acc, &p| acc + ((p < r) as usize));
        ixs.push(ct);
    }
    ixs
}
