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

    pub fn from_vec(asgn: Vec<usize>, alpha: f64) -> Self {
        let ncats: usize = *asgn.iter().max().unwrap() + 1;
        let mut counts: Vec<usize> = vec![0; ncats];
        for &z in asgn.iter() {
            counts[z] += 1;
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
        for w in weights.iter_mut() {
            *w /= z;
        }
        weights
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


#[cfg(test)]
mod tests {
    use super::*;
    use self::rand::XorShiftRng;


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
}
