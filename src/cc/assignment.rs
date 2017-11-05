extern crate rand;

use self::rand::Rng;
use misc::pflip;


#[allow(dead_code)]
pub struct Assignment {
    pub alpha: f64,
    pub asgn: Vec<usize>,
    pub counts: Vec<usize>,
    pub ncats: usize,
}

impl Assignment {
    pub fn draw<R: Rng>(n: usize, alpha: f64, rng: &mut R) -> Assignment {
        let mut ncats = 1;
        let mut weights: Vec<f64> = vec![1.0];
        let mut asgn: Vec<usize> = Vec::with_capacity(n);

        asgn.push(0);

        for _ in 1..n {
            weights.push(alpha);
            let k = pflip(&weights, rng);
            asgn.push(k);

            if k == ncats {
                weights[ncats] = 1.0;
                ncats += 1;
            } else {
                weights.truncate(ncats);
                weights[k] += 1.0;
            }
        }
        // convert weights to counts, correcting for possible floating point
        // errors
        let counts: Vec<usize> = weights.iter()
                                         .map(|w| (w + 0.5) as usize)
                                         .collect();

        Assignment{alpha: alpha, asgn: asgn, counts: counts, ncats: ncats}
    }

    pub fn flat(n: usize, alpha: f64) -> Assignment {
        let asgn: Vec<usize> = vec![0; n];
        let counts: Vec<usize> = vec![n];
        Assignment{alpha: alpha, asgn: asgn, counts: counts, ncats: 1}
    }
}
