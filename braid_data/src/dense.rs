use crate::{AccumScore, Container};
use braid_stats::Datum;
use std::convert::TryFrom;

/// The baseline data structure
pub struct DenseContainer<T: Clone> {
    /// The actual values of the data. Uses `Default::default()`
    pub values: Vec<T>,
    /// Tells whether each values is present or missing. BitVec takes up 1/8th
    /// the memory of Vec<bool>.
    pub present: Vec<bool>,
}

impl<T: Clone> Default for DenseContainer<T> {
    fn default() -> DenseContainer<T> {
        DenseContainer {
            values: vec![],
            present: vec![],
        }
    }
}

impl<T: Clone + Default> DenseContainer<T> {
    pub fn new(values: Vec<T>, present: Vec<bool>) -> DenseContainer<T> {
        DenseContainer {
            values,
            present: {
                let mut bv = Vec::with_capacity(present.len());
                present.iter().for_each(|&b| bv.push(b));
                bv
            },
        }
    }
}

impl<T: Clone + Default + TryFrom<Datum>> Container<T> for DenseContainer<T> {
    fn get_slices(&self) -> Vec<(usize, &[T])> {
        vec![(0, self.values.as_slice())]
    }

    fn get(&self, ix: usize) -> Option<T> {
        if self.present[ix] {
            Some(self.values[ix].clone())
        } else {
            None
        }
    }

    fn present_cloned(&self) -> Vec<T> {
        unimplemented!()
    }

    fn insert_overwrite(&mut self, ix: usize, x: T) {
        if ix >= self.values.len() {
            self.values.resize_with(ix + 1, Default::default);
            self.present.resize_with(ix + 1, Default::default);
            self.present[ix] = true;
        } else {
            self.values[ix] = x;
            self.present[ix] = true;
        }
    }

    fn remove(&mut self, ix: usize) -> Option<T> {
        if self.present[ix] {
            let out = self.values[ix].clone();
            self.values[ix] = T::default();
            self.present[ix] = false;
            Some(out)
        } else {
            None
        }
    }

    fn push(&mut self, xopt: Option<T>) {
        match xopt {
            None => {
                self.present.push(false);
                self.values.push(T::default());
            }
            Some(x) => {
                self.present.push(true);
                self.values.push(x);
            }
        }
    }
}

impl<T: Clone + Default> AccumScore<T> for DenseContainer<T> {
    fn accum_score<F: Fn(&T) -> f64>(&self, scores: &mut [f64], ln_f: &F) {
        self.values
            .iter()
            .zip(self.present.iter())
            .zip(scores.iter_mut())
            .for_each(|((x, &pr), y)| {
                if pr {
                    *y += ln_f(x);
                }
            })
    }
}
