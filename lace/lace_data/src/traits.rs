use crate::Datum;
use std::convert::TryFrom;

pub trait AccumScore<T> {
    /// Compute scores on the data using `score_fn` and add them to `scores`
    fn accum_score<F: Fn(&T) -> f64>(&self, scores: &mut [f64], score_fn: &F);
}

/// A data container
pub trait Container<T: Clone + TryFrom<Datum>> {
    /// get the data slices and the start indices
    fn get_slices(&self) -> Vec<(usize, &[T])>;

    /// Get the entry at ix if it exists
    fn get(&self, ix: usize) -> Option<T>;

    /// Insert or overwrite an entry at ix
    fn insert_overwrite(&mut self, ix: usize, x: T);

    /// Append a new datum to the end of the container
    fn push(&mut self, xopt: Option<T>);

    /// Append n new empty entries to the container
    fn upsize(&mut self, n: usize) {
        (0..n).for_each(|_| self.push(None));
    }

    /// Get as cloned vector containing only the present data
    fn present_cloned(&self) -> Vec<T>;

    /// Remove and return the entry at ix if it exists. Used to mark a present
    /// datum as missing, not to completely remove a record. Does not decrease
    /// the length.
    fn remove(&mut self, ix: usize) -> Option<T>;

    // TODO: should return result type
    fn push_datum(&mut self, x: Datum) {
        match x {
            Datum::Missing => self.push(None),
            _ => {
                if let Ok(val) = T::try_from(x) {
                    self.push(Some(val));
                } else {
                    panic!("failed to convert pushed datum");
                }
            }
        }
    }

    fn insert_datum(&mut self, row_ix: usize, x: Datum) {
        match x {
            Datum::Missing => {
                self.remove(row_ix);
            }
            _ => {
                if let Ok(val) = T::try_from(x) {
                    self.insert_overwrite(row_ix, val)
                } else {
                    panic!("failed to convert inserted datum");
                }
            }
        }
    }
}
