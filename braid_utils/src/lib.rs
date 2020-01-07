mod matrix;
mod misc;
pub mod numbers;
mod random;
mod stats;
mod unique;

pub use matrix::*;
pub use misc::*;
pub use random::*;
pub use stats::*;
pub use unique::*;

/// Iterates over a set of categorical ranges
pub struct CategoricalCartProd {
    ranges: Vec<u8>,
    item: Vec<u8>,
    start: bool,
}

impl CategoricalCartProd {
    // FIXME: Will fail if ranges.len() is 1
    pub fn new(ranges: Vec<u8>) -> Self {
        CategoricalCartProd {
            item: vec![0; ranges.len()],
            ranges,
            start: true,
        }
    }
}

impl Iterator for CategoricalCartProd {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start {
            self.start = false;
            return Some(self.item.clone());
        }

        let mut k = self.item.len() - 1;

        if self.item[k] == self.ranges[k] - 1 {
            loop {
                self.item[k] = 0;
                k -= 1;
                if self.item[k] != (self.ranges[k] - 1) {
                    self.item[k] += 1;
                    return Some(self.item.clone());
                } else if k == 0 {
                    return None;
                }
            }
        } else {
            self.item[k] += 1;
            Some(self.item.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn cartprod_single() {
    //     let mut cartprod = CategoricalCartProd::new(vec![3]);
    //     assert_eq!(cartprod.next(), Some(vec![0]));
    //     assert_eq!(cartprod.next(), Some(vec![1]));
    //     assert_eq!(cartprod.next(), Some(vec![2]));
    //     assert_eq!(cartprod.next(), None);
    // }

    #[test]
    fn cartprod_dual() {
        let mut cartprod = CategoricalCartProd::new(vec![3, 2]);
        assert_eq!(cartprod.next(), Some(vec![0, 0]));
        assert_eq!(cartprod.next(), Some(vec![0, 1]));
        assert_eq!(cartprod.next(), Some(vec![1, 0]));
        assert_eq!(cartprod.next(), Some(vec![1, 1]));
        assert_eq!(cartprod.next(), Some(vec![2, 0]));
        assert_eq!(cartprod.next(), Some(vec![2, 1]));
        assert_eq!(cartprod.next(), None);
    }

    #[test]
    fn cartprod_triple() {
        let mut cartprod = CategoricalCartProd::new(vec![3, 2, 1]);
        assert_eq!(cartprod.next(), Some(vec![0, 0, 0]));
        assert_eq!(cartprod.next(), Some(vec![0, 1, 0]));
        assert_eq!(cartprod.next(), Some(vec![1, 0, 0]));
        assert_eq!(cartprod.next(), Some(vec![1, 1, 0]));
        assert_eq!(cartprod.next(), Some(vec![2, 0, 0]));
        assert_eq!(cartprod.next(), Some(vec![2, 1, 0]));
        assert_eq!(cartprod.next(), None);
    }
}
