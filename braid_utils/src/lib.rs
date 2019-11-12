pub mod misc;
pub mod numbers;
pub mod random;
pub mod stats;
pub mod unique;

/// Iterates over a set of categorical ranges
pub struct CategoricalCartProd {
    ranges: Vec<u8>,
    item: Vec<u8>,
    start: bool,
}

impl CategoricalCartProd {
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
