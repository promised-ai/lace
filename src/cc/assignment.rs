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
    pub fn draw<R: Rng>(n: usize, alpha: f64, rng: &mut R) -> Self {
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

    pub fn flat(n: usize, alpha: f64) -> Self {
        let asgn: Vec<usize> = vec![0; n];
        let counts: Vec<usize> = vec![n];
        Assignment{alpha: alpha, asgn: asgn, counts: counts, ncats: 1}
    }

    pub fn from_vec(alpha: f64, asgn: Vec<usize>) -> Self {
        let ncats: usize = *asgn.iter().max().unwrap();
        let mut counts: Vec<usize> = vec![0; ncats];
        for &z in asgn.iter() {
            counts[z] += 1;
        }
        Assignment{alpha: alpha, asgn: asgn, counts: counts, ncats: ncats}
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use self::rand::XorShiftRng;


    #[test]
    fn drawn_assignment_should_have_valid_partition() {
        let n: usize = 50;
        let alpha: f64 = 1.0;
        let mut rng = XorShiftRng::new_unseeded();

        // do the test 100 times because it's random
        for _ in 0..100 {
            let asgn = Assignment::draw(n, alpha, &mut rng);
            let max_ix = *asgn.asgn.iter().max().unwrap();
            let min_ix = *asgn.asgn.iter().min().unwrap();

            assert_eq!(asgn.counts.len(), asgn.ncats);
            assert_eq!(asgn.counts.len(), max_ix + 1);
            assert_eq!(min_ix, 0);

            for (k, &count) in asgn.counts.iter().enumerate() {
                let k_count = asgn.asgn.iter().fold(0, |acc, &z| {
                    if z == k {acc + 1} else {acc}
                });
                assert_eq!(k_count, count);
            }
        }
    }


    #[test]
    fn flat_partition_validation() {
        let n: usize = 50;
        let alpha: f64 = 1.0;

        let asgn = Assignment::flat(n, alpha);

        assert_eq!(asgn.ncats, 1);
        assert_eq!(asgn.counts.len(), 1);
        assert_eq!(asgn.counts[0], n);
        assert!(asgn.asgn.iter().all(|&z| z == 0));
    }
}
