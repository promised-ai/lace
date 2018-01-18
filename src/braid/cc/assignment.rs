extern crate rand;

use self::rand::Rng;
use self::rand::distributions::{Gamma, IndependentSample};
use misc::pflip;
use misc::mh::mh_prior;
use special::gammaln;


#[allow(dead_code)]
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct Assignment {
    pub alpha: f64,
    pub asgn: Vec<usize>,
    pub counts: Vec<usize>,
    pub ncats: usize,
}


pub struct AssignmentDiagnostics {
    asgn_min_is_zero: bool,
    asgn_max_is_ncats_minus_one: bool,
    asgn_contains_0_through_ncats_minus_1: bool,
    no_zero_counts: bool,
    ncats_cmp_counts_len: bool,
    sum_counts_cmp_n: bool,
    asgn_agrees_with_counts: bool,
}


impl AssignmentDiagnostics {
    pub fn is_valid(&self) -> bool {
        self.asgn_min_is_zero
            && self.asgn_max_is_ncats_minus_one
            && self.asgn_contains_0_through_ncats_minus_1
            && self.no_zero_counts
            && self.ncats_cmp_counts_len
            && self.sum_counts_cmp_n
            && self.asgn_agrees_with_counts
    }
}


impl Assignment {
    /// Draws alpha from prior then draws an n-length partition
    pub fn from_prior<R: Rng>(n: usize, mut rng: &mut R) -> Self {
        let prior = Gamma::new(1.0, 1.0);  // inverse of prior
        let alpha = 1.0 / prior.ind_sample(&mut rng);
        Self::draw(n, alpha, &mut rng)
    }

    /// Draws an n-length assignment from CPR(alpha)
    pub fn draw<R: Rng>(n: usize, alpha: f64, rng: &mut R) -> Self {
        let mut ncats = 1;
        let mut weights: Vec<f64> = vec![1.0];
        let mut asgn: Vec<usize> = Vec::with_capacity(n);

        asgn.push(0);

        for _ in 1..n {
            weights.push(alpha);
            let k = pflip(&weights, 1, rng)[0];
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

    /// Creates an assignment with one category
    pub fn flat(n: usize, alpha: f64) -> Self {
        let asgn: Vec<usize> = vec![0; n];
        let counts: Vec<usize> = vec![n];
        Assignment{alpha: alpha, asgn: asgn, counts: counts, ncats: 1}
    }

    /// Creates an assignment with `ncats` uniformly-occupied categories
    pub fn with_ncats(n: usize, ncats: usize, alpha: f64) -> Self {
        if ncats > n {
            panic!("n must be greater than ncats");
        }
        if ncats == 0 {
            panic!("ncats must be greater than 0");
        }
        let asgn_vec: Vec<usize> = (0..n).map(|i| i % ncats).collect();
        Assignment::from_vec(asgn_vec, alpha)
    }

    pub fn len(&self) -> usize {
        self.asgn.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn from_vec(asgn: Vec<usize>, alpha: f64) -> Self {
        let ncats: usize = *asgn.iter().max().unwrap() + 1;
        let mut counts: Vec<usize> = vec![0; ncats];
        for z in &asgn {
            counts[*z] += 1;
        }
        Assignment{alpha: alpha, asgn: asgn, counts: counts, ncats: ncats}
    }

    pub fn dirvec(&self, append_alpha: bool) -> Vec<f64> {
        let mut dv: Vec<f64> = self.counts.iter().map(|&x| x as f64).collect();
        if append_alpha {
            dv.push(self.alpha);
        }
        dv
    }

    pub fn weights(&self) -> Vec<f64> {
        let mut weights = self.dirvec(false);
        let z: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= z;
        }
        weights
    }

    pub fn log_weights(&self) -> Vec<f64> {
        self.weights().iter().map(|w| w.ln()).collect()
    }

    pub fn update_alpha(&mut self, n_iter: usize, mut rng: &mut Rng) {
        let cts = &self.counts;
        let n: usize = self.len();
        let loglike = |alpha: &f64| { lcrp(n, cts, *alpha) };
        let prior = Gamma::new(1.0, 1.0); // inverse of prior
        let prior_draw = |mut rng: &mut Rng| {
            1.0 / prior.ind_sample(&mut rng) 
        };
        self.alpha = mh_prior(loglike, prior_draw, n_iter, &mut rng);
    }

    pub fn validate(&self) -> AssignmentDiagnostics {
        AssignmentDiagnostics {
            asgn_min_is_zero: {
                *self.asgn.iter().min().unwrap() == 0
            },
            asgn_max_is_ncats_minus_one: {
                *self.asgn.iter().max().unwrap() == self.ncats - 1
            },
            asgn_contains_0_through_ncats_minus_1: {
                let mut so_far = true;
                for k in 0..self.ncats {
                    so_far = so_far && self.asgn.iter().any(|&x| x == k)
                }
                so_far
            },
            no_zero_counts: {
                !self.counts.iter().any(|&ct| ct == 0)
            },
            ncats_cmp_counts_len: {
                self.ncats == self.counts.len()
            },
            sum_counts_cmp_n: {
                let n: usize = self.counts.iter().sum();
                n == self.asgn.len()
            },
            asgn_agrees_with_counts: {
                let mut all = true;
                for (k, &count) in self.counts.iter().enumerate() {
                    let k_count = self.asgn.iter().fold(0, |acc, &z| {
                        if z == k {acc + 1} else {acc}
                    });
                    all = all && (k_count == count)
                }
                all
            }
        }
    }
}


fn lcrp(n: usize, cts: &[usize], alpha: f64) -> f64 {
    let k: f64 = cts.len() as f64;
    let gsum = cts.iter().fold(0.0, |acc, ct| { acc + gammaln(*ct as f64) });
    gsum + k*alpha.ln() + gammaln(alpha) - gammaln(n as f64 + alpha)
}


#[cfg(test)]
mod tests {
    extern crate serde_test;

    use super::*;
    use self::rand::XorShiftRng;
    use self::serde_test::{Token, assert_tokens};


    #[test]
    fn zero_count_fails_validation() {
        let asgn = Assignment{
            alpha: 1.0,
            asgn: vec![0, 0, 0, 0],
            counts: vec![0, 4],
            ncats: 1,
        };

        let diagnostic = asgn.validate();

        assert!(!diagnostic.is_valid());

        assert!(diagnostic.asgn_min_is_zero);
        assert!(diagnostic.asgn_max_is_ncats_minus_one);
        assert!(diagnostic.asgn_contains_0_through_ncats_minus_1);
        assert!(diagnostic.sum_counts_cmp_n);
        assert!(!diagnostic.ncats_cmp_counts_len);
        assert!(!diagnostic.no_zero_counts);
        assert!(!diagnostic.asgn_agrees_with_counts);
    }


    #[test]
    fn bad_counts_fails_validation() {
        let asgn = Assignment{
            alpha: 1.0,
            asgn: vec![1, 1, 0, 0],
            counts: vec![2, 3],
            ncats: 2,
        };

        let diagnostic = asgn.validate();

        assert!(!diagnostic.is_valid());

        assert!(diagnostic.asgn_min_is_zero);
        assert!(diagnostic.asgn_max_is_ncats_minus_one);
        assert!(diagnostic.asgn_contains_0_through_ncats_minus_1);
        assert!(!diagnostic.sum_counts_cmp_n);
        assert!(diagnostic.ncats_cmp_counts_len);
        assert!(diagnostic.no_zero_counts);
        assert!(!diagnostic.asgn_agrees_with_counts);
    }


    #[test]
    fn low_ncats_fails_validation() {
        let asgn = Assignment{
            alpha: 1.0,
            asgn: vec![1, 1, 0, 0],
            counts: vec![2, 2],
            ncats: 1,
        };

        let diagnostic = asgn.validate();

        assert!(!diagnostic.is_valid());

        assert!(diagnostic.asgn_min_is_zero);
        assert!(!diagnostic.asgn_max_is_ncats_minus_one);
        assert!(diagnostic.asgn_contains_0_through_ncats_minus_1);
        assert!(diagnostic.sum_counts_cmp_n);
        assert!(!diagnostic.ncats_cmp_counts_len);
        assert!(diagnostic.no_zero_counts);
        assert!(diagnostic.asgn_agrees_with_counts);
    }


    #[test]
    fn high_ncats_fails_validation() {
        let asgn = Assignment{
            alpha: 1.0,
            asgn: vec![1, 1, 0, 0],
            counts: vec![2, 2],
            ncats: 3,
        };

        let diagnostic = asgn.validate();

        assert!(!diagnostic.is_valid());

        assert!(diagnostic.asgn_min_is_zero);
        assert!(!diagnostic.asgn_max_is_ncats_minus_one);
        assert!(!diagnostic.asgn_contains_0_through_ncats_minus_1);
        assert!(diagnostic.sum_counts_cmp_n);
        assert!(!diagnostic.ncats_cmp_counts_len);
        assert!(diagnostic.no_zero_counts);
        assert!(diagnostic.asgn_agrees_with_counts);
    }

    #[test]
    fn no_zero_cat_fails_validation() {
        let asgn = Assignment{
            alpha: 1.0,
            asgn: vec![1, 1, 2, 2],
            counts: vec![2, 2],
            ncats: 2,
        };

        let diagnostic = asgn.validate();

        assert!(!diagnostic.is_valid());

        assert!(!diagnostic.asgn_min_is_zero);
        assert!(!diagnostic.asgn_max_is_ncats_minus_one);
        assert!(!diagnostic.asgn_contains_0_through_ncats_minus_1);
        assert!(diagnostic.sum_counts_cmp_n);
        assert!(diagnostic.ncats_cmp_counts_len);
        assert!(diagnostic.no_zero_counts);
        assert!(!diagnostic.asgn_agrees_with_counts);
    }

    #[test]
    fn drawn_assignment_should_have_valid_partition() {
        let n: usize = 50;
        let alpha: f64 = 1.0;
        let mut rng = XorShiftRng::new_unseeded();

        // do the test 100 times because it's random
        for _ in 0..100 {
            let asgn = Assignment::draw(n, alpha, &mut rng);
            assert!(asgn.validate().is_valid());
        }
    }

    #[test]
    fn from_prior_shouls_have_valid_alpha_and_proper_length() {
        let n: usize = 50;
        let mut rng = XorShiftRng::new_unseeded();
        let asgn = Assignment::from_prior(n, &mut rng);

        assert!(!asgn.is_empty());
        assert_eq!(asgn.len(), n);
        assert!(asgn.validate().is_valid());
        assert!(asgn.alpha > 0.0);
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


    #[test]
    fn from_vec() {
        let asgn = Assignment::from_vec(vec![0, 1, 2, 0, 1, 0], 1.0);
        assert_eq!(asgn.ncats, 3);
        assert_eq!(asgn.counts[0], 3);
        assert_eq!(asgn.counts[1], 2);
        assert_eq!(asgn.counts[2], 1);
    }


    #[test]
    fn with_ncats_ncats_evenly_divides_n() {
        let asgn = Assignment::with_ncats(100, 5, 1.0);
        assert!(asgn.validate().is_valid());
        assert_eq!(asgn.ncats, 5);
        assert_eq!(asgn.counts[0], 20);
        assert_eq!(asgn.counts[1], 20);
        assert_eq!(asgn.counts[2], 20);
        assert_eq!(asgn.counts[3], 20);
        assert_eq!(asgn.counts[4], 20);
    }


    #[test]
    fn with_ncats_ncats_doesnt_divides_n() {
        let asgn = Assignment::with_ncats(103, 5, 1.0);
        assert!(asgn.validate().is_valid());
        assert_eq!(asgn.ncats, 5);
        assert_eq!(asgn.counts[0], 21);
        assert_eq!(asgn.counts[1], 21);
        assert_eq!(asgn.counts[2], 21);
        assert_eq!(asgn.counts[3], 20);
        assert_eq!(asgn.counts[4], 20);
    }


    #[test]
    fn dirvec_no_alpha() {
        let asgn = Assignment::from_vec(vec![0, 1, 2, 0, 1, 0], 1.0);
        let dv = asgn.dirvec(false);

        assert_eq!(dv.len(), 3);
        assert_relative_eq!(dv[0], 3.0, epsilon = 10E-10);
        assert_relative_eq!(dv[1], 2.0, epsilon = 10E-10);
        assert_relative_eq!(dv[2], 1.0, epsilon = 10E-10);
    }


    #[test]
    fn dirvec_with_alpha() {
        let asgn = Assignment::from_vec(vec![0, 1, 2, 0, 1, 0], 1.5);
        let dv = asgn.dirvec(true);

        assert_eq!(dv.len(), 4);
        assert_relative_eq!(dv[0], 3.0, epsilon = 10E-10);
        assert_relative_eq!(dv[1], 2.0, epsilon = 10E-10);
        assert_relative_eq!(dv[2], 1.0, epsilon = 10E-10);
        assert_relative_eq!(dv[3], 1.5, epsilon = 10E-10);
    }


    #[test]
    fn weights() {
        let asgn = Assignment::from_vec(vec![0, 1, 2, 0, 1, 0], 1.0);
        let weights = asgn.weights();

        assert_eq!(weights.len(), 3);
        assert_relative_eq!(weights[0], 3.0/6.0, epsilon = 10E-10);
        assert_relative_eq!(weights[1], 2.0/6.0, epsilon = 10E-10);
        assert_relative_eq!(weights[2], 1.0/6.0, epsilon = 10E-10);
    }

    #[test]
    fn serialize_and_deserialize() {
        let asgn = Assignment::from_vec(vec![0, 1, 1], 1.0);
        assert_tokens(&asgn, &[
            Token::Struct { name: "Assignment", len: 4 },
            Token::Str("alpha"),
            Token::F64(1.0),
            Token::Str("asgn"),
            Token::Seq { len: Some(3) },
            Token::U64(0),
            Token::U64(1),
            Token::U64(1),
            Token::SeqEnd,
            Token::Str("counts"),
            Token::Seq { len: Some(2) },
            Token::U64(1),
            Token::U64(2),
            Token::SeqEnd,
            Token::Str("ncats"),
            Token::U64(2),
            Token::StructEnd,
        ]);
    }

    #[test]
    fn lcrp_all_ones() {
        let lcrp_1 = lcrp(4, &vec![1, 1, 1, 1], 1.0);
        assert_relative_eq!(lcrp_1, -3.17805383034795, epsilon=10E-8);

        let lcrp_2 = lcrp(4, &vec![1, 1, 1, 1], 2.1);
        assert_relative_eq!(lcrp_2, -1.94581759074351, epsilon=10E-8);
    }
}
