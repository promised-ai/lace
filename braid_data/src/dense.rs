use crate::{AccumScore, Container};
use braid_stats::Datum;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

/// The baseline data structure
#[derive(Clone, Serialize, Deserialize)]
pub struct DenseContainer<T> {
    /// Tells whether each values is present or missing. BitVec takes up 1/8th
    /// the memory of Vec<bool>.
    pub present: Vec<bool>,
    /// The actual values of the data. Uses `Default::default()`
    pub data: Vec<T>,
}

impl<T: Clone> Default for DenseContainer<T> {
    fn default() -> DenseContainer<T> {
        DenseContainer {
            data: vec![],
            present: vec![],
        }
    }
}

impl<T: Clone + Default> DenseContainer<T> {
    pub fn new(data: Vec<T>, present: Vec<bool>) -> DenseContainer<T> {
        DenseContainer {
            data,
            present: {
                let mut bv = Vec::with_capacity(present.len());
                present.iter().for_each(|&b| bv.push(b));
                bv
            },
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Clone + Default + TryFrom<Datum>> Container<T> for DenseContainer<T> {
    fn get_slices(&self) -> Vec<(usize, &[T])> {
        vec![(0, self.data.as_slice())]
    }

    fn get(&self, ix: usize) -> Option<T> {
        if self.present[ix] {
            Some(self.data[ix].clone())
        } else {
            None
        }
    }

    fn present_cloned(&self) -> Vec<T> {
        unimplemented!()
    }

    fn insert_overwrite(&mut self, ix: usize, x: T) {
        if ix >= self.data.len() {
            self.data.resize_with(ix + 1, Default::default);
            self.present.resize_with(ix + 1, Default::default);
            self.present[ix] = true;
        } else {
            self.data[ix] = x;
            self.present[ix] = true;
        }
    }

    fn remove(&mut self, ix: usize) -> Option<T> {
        if self.present[ix] {
            let out = self.data[ix].clone();
            self.data[ix] = T::default();
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
                self.data.push(T::default());
            }
            Some(x) => {
                self.present.push(true);
                self.data.push(x);
            }
        }
    }
}

impl<T: Clone + Default> AccumScore<T> for DenseContainer<T> {
    fn accum_score<F: Fn(&T) -> f64>(&self, scores: &mut [f64], ln_f: &F) {
        self.data
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
