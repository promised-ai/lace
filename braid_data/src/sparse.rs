use crate::{AccumScore, Container, DenseContainer};
use braid_stats::Datum;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

// TODO: Default trait bound exists only for serde from/into.
/// A sparse container stores contiguous vertical slices of data
#[serde(from = "DenseContainer<T>")]
#[serde(into = "DenseContainer<T>")]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SparseContainer<T: Clone + Default> {
    n: usize,
    /// Each entry is the index of the index of the first entry
    pub(crate) data: Vec<(usize, Vec<T>)>,
}

// TODO: Impl Error for this
#[derive(Clone, Debug)]
pub struct ValuesPresentMismatchError {
    pub values_len: usize,
    pub present_len: usize,
}

impl<T: Clone + Default> SparseContainer<T> {
    pub fn new() -> SparseContainer<T> {
        SparseContainer { n: 0, data: vec![] }
    }

    /// create an n-length container will all missing data
    pub fn all_missing(n: usize) -> SparseContainer<T> {
        SparseContainer { n, data: vec![] }
    }

    pub fn present_iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter().map(|(_, xs)| xs).flatten()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Ensure all adjacent data slices are joined. Reduces indirection.
    /// Returns the number of slice merge operations performed.
    pub fn defragment(&mut self) -> usize {
        let mut slice_ix = 0;
        let mut n_merged = 0;

        while slice_ix < self.data.len() - 1 {
            if self.check_merge_next(slice_ix) {
                n_merged += 1;
            }
            slice_ix += 1;
        }
        n_merged
    }

    /// Determines whether an insert joined two data slices and merges them
    /// internally if so.
    ///
    /// Returns `true` if the slices at `slice_ix` and `slice_ix + 1` were
    /// adjacent and merged.
    fn check_merge_next(&mut self, slice_ix: usize) -> bool {
        if slice_ix == self.data.len() - 1 {
            return false;
        }

        let start_ix = self.data[slice_ix].0 + self.data[slice_ix].1.len();
        if start_ix == self.data[slice_ix + 1].0 {
            let (_, mut bottom) = self.data.remove(slice_ix + 1);
            self.data[slice_ix].1.append(&mut bottom);
            true
        } else {
            false
        }
    }
}

impl<T: Clone + TryFrom<Datum> + Default> Container<T> for SparseContainer<T> {
    fn get_slices(&self) -> Vec<(usize, &[T])> {
        self.data
            .iter()
            .map(|(ix, xs)| (*ix, xs.as_slice()))
            .collect()
    }

    fn get(&self, ix: usize) -> Option<T> {
        if ix >= self.n {
            panic!("out of bounds: ix was {} but len is {}", ix, self.len());
        } else if self.data.is_empty() {
            None
        } else if self.data[0].0 > ix {
            None
        } else {
            let result = self.data.binary_search_by(|entry| entry.0.cmp(&ix));

            match result {
                Ok(index) => Some(self.data[index].1[0].clone()),
                Err(index) => {
                    let n = self.data[index - 1].1.len();
                    let start_ix = self.data[index - 1].0;
                    let local_ix = ix - start_ix;

                    if ix >= start_ix + n {
                        // Between data slices. Missing data.
                        None
                    } else {
                        Some(self.data[index - 1].1[local_ix].clone())
                    }
                }
            }
        }
    }

    fn present_cloned(&self) -> Vec<T> {
        let mut present_data = Vec::new();
        self.data.iter().for_each(|(_, xs)| {
            present_data.extend_from_slice(xs);
        });
        present_data
    }

    fn push(&mut self, xopt: Option<T>) {
        if let Some(x) = xopt {
            match self.data.last_mut() {
                Some(entry) => {
                    let last_occupied_ix = entry.0 + entry.1.len();
                    if last_occupied_ix < self.n + 1 {
                        self.data.push((self.len(), vec![x]));
                        self.n += 1;
                    } else if last_occupied_ix == self.n + 1 {
                        self.n += 1;
                        entry.1.push(x);
                    } else {
                        // if we bookkeep correctly, we should never get here
                        panic!(
                            "last occupied index ({}) greater than n ({})",
                            last_occupied_ix, self.n
                        )
                    }
                }
                None => {
                    assert!(self.data.is_empty());
                    self.data.push((self.n, vec![x]));
                    self.n += 1;
                }
            }
        } else {
            self.n += 1;
        }
    }

    // FIXME: put merge checks back in
    fn insert_overwrite(&mut self, ix: usize, x: T) {
        let result = self.data.binary_search_by(|entry| entry.0.cmp(&ix));
        match result {
            Ok(index) => {
                self.data[index].1[0] = x;
            }
            Err(index) => {
                if index == 0 {
                    if self.data.is_empty() {
                        self.data.push((ix, vec![x]));
                    } else {
                        let data_0 = &mut self.data[0];
                        // data is missing and before any existing data
                        if ix == data_0.0 - 1 {
                            // inserted datum sits on top of the top slice
                            data_0.0 = ix;
                            data_0.1.insert(0, x);
                        } else {
                            // inserted data is not contiguous with top data
                            // TODO: better way to insert at index 0?
                            self.data.insert(0, (ix, vec![x]));
                            self.check_merge_next(index);
                        }
                    }
                    // The new datum was inserted outside vector, so increase the count
                    if ix >= self.n {
                        self.n = ix + 1;
                    }
                } else {
                    let start_ix = self.data[index - 1].0;
                    let data = &mut self.data[index - 1].1;
                    let n = data.len();

                    let end_ix = start_ix + n;

                    // check if present
                    let present = ix < end_ix;

                    if present {
                        let local_ix = ix - start_ix;
                        data[local_ix] = x;
                    } else {
                        if ix == end_ix {
                            data.push(x);
                        } else {
                            self.data.insert(index, (ix, vec![x]));
                        }
                    }
                    self.check_merge_next(index - 1);

                    // The new datum was inserted outside vector, so increase the count
                    if ix >= self.n {
                        self.n = ix + 1;
                    }
                }
            }
        }
    }

    fn remove(&mut self, ix: usize) -> Option<T> {
        let result = self.data.binary_search_by(|entry| entry.0.cmp(&ix));
        match result {
            Ok(index) => {
                let val = self.data[index].1.remove(0);
                self.data[index].0 += 1;
                Some(val)
            }
            Err(index) => {
                if index == 0 {
                    None
                } else {
                    let n = self.data[index - 1].1.len();
                    let start_ix = self.data[index - 1].0;
                    let local_ix = ix - start_ix;

                    if ix >= start_ix + n {
                        // Between data slices. Missing data.
                        None
                    } else if ix == start_ix + n - 1 {
                        // pop returns option, but this should never be none.
                        // We could 'check' by unwrapping and then wrapping the
                        // result in `Some`, but that seems wasteful.
                        self.data[index - 1].1.pop()
                    } else {
                        // have to remove and split
                        let tail =
                            self.data[index - 1].1.split_off(local_ix + 1);
                        self.data.insert(index, (ix + 1, tail));

                        // same situation with pop as in previous branch
                        self.data[index - 1].1.pop()
                    }
                }
            }
        }
    }
}

impl<T: Clone + Default> AccumScore<T> for SparseContainer<T> {
    fn accum_score<F: Fn(&T) -> f64>(&self, scores: &mut [f64], ln_f: &F) {
        self.data.iter().for_each(|(ix, xs)| {
            // XXX: Getting the sub-slices here allows us to use iterators which
            // bypasses bounds checking when x[i] is called. Bounds checking slows
            // things down considerably.
            let target_sub = unsafe {
                let ptr = scores.as_mut_ptr().add(*ix);
                std::slice::from_raw_parts_mut(ptr, xs.len())
            };

            target_sub.iter_mut().zip(xs.iter()).for_each(|(y, x)| {
                *y += ln_f(x);
            })
        })
    }
}

impl<T: Clone + Default> From<Vec<T>> for SparseContainer<T> {
    fn from(xs: Vec<T>) -> SparseContainer<T> {
        if xs.is_empty() {
            SparseContainer::new()
        } else {
            SparseContainer {
                n: xs.len(),
                data: vec![(0, xs)],
            }
        }
    }
}

impl<T: Clone + Default> From<Vec<(T, bool)>> for SparseContainer<T> {
    fn from(mut xs: Vec<(T, bool)>) -> SparseContainer<T> {
        if xs.is_empty() {
            SparseContainer::new()
        } else {
            let n = xs.len();
            let mut data: Vec<(usize, Vec<T>)> = Vec::new();
            let mut filling: bool = false;

            for (i, (x, pr)) in xs.drain(..).enumerate() {
                if filling {
                    if pr {
                        // push to last data vec
                        data.last_mut().unwrap().1.push(x);
                    } else {
                        // stop filling
                        filling = false;
                    }
                } else if pr {
                    // create a new data vec and start filling
                    data.push((i, vec![x]));
                    filling = true;
                }
            }

            SparseContainer { data, n }
        }
    }
}

impl<T: Clone + Default> Default for SparseContainer<T> {
    fn default() -> SparseContainer<T> {
        SparseContainer::new()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::DenseContainer;

    fn sparse_container() -> SparseContainer<f64> {
        let xs: Vec<(f64, bool)> = vec![
            (0.0, false),
            (0.0, false),
            (1.0, true),
            (2.0, true),
            (3.0, true),
            (0.0, false),
            (0.0, false),
            (1.0, true),
            (2.0, true),
            (3.0, true),
            (4.0, true),
        ];
        SparseContainer::from(xs)
    }

    #[test]
    fn from_dense_all_populated() {
        let dense: DenseContainer<u8> =
            DenseContainer::new(vec![0, 1, 2, 3, 4], vec![true; 5]);
        let sparse = SparseContainer::from(dense);
        assert_eq!(sparse.len(), 5);
        assert_eq!(sparse.get(0).unwrap(), 0);
        assert_eq!(sparse.get(1).unwrap(), 1);
        assert_eq!(sparse.get(2).unwrap(), 2);
        assert_eq!(sparse.get(3).unwrap(), 3);
        assert_eq!(sparse.get(4).unwrap(), 4);
    }

    #[test]
    fn from_dense_head_missing() {
        let dense: DenseContainer<u8> = DenseContainer::new(
            vec![0, 1, 2, 3, 4],
            vec![false, false, true, true, true],
        );
        let sparse = SparseContainer::from(dense);
        assert_eq!(sparse.len(), 5);
        assert_eq!(sparse.get(0), None);
        assert_eq!(sparse.get(1), None);
        assert_eq!(sparse.get(2), Some(2));
        assert_eq!(sparse.get(3), Some(3));
        assert_eq!(sparse.get(4), Some(4));
    }

    #[test]
    fn from_dense_tail_missing() {
        let dense: DenseContainer<u8> = DenseContainer::new(
            vec![0, 1, 2, 3, 4],
            vec![true, true, true, false, false],
        );
        let sparse = SparseContainer::from(dense);
        assert_eq!(sparse.len(), 5);
        assert_eq!(sparse.get(0), Some(0));
        assert_eq!(sparse.get(1), Some(1));
        assert_eq!(sparse.get(2), Some(2));
        assert_eq!(sparse.get(3), None);
        assert_eq!(sparse.get(4), None);
    }

    #[test]
    fn from_dense_middle_missing() {
        let dense: DenseContainer<u8> = DenseContainer::new(
            vec![0, 1, 2, 3, 4],
            vec![true, true, false, false, true],
        );
        let sparse = SparseContainer::from(dense);
        assert_eq!(sparse.len(), 5);
        assert_eq!(sparse.get(0), Some(0));
        assert_eq!(sparse.get(1), Some(1));
        assert_eq!(sparse.get(2), None);
        assert_eq!(sparse.get(3), None);
        assert_eq!(sparse.get(4), Some(4));
    }

    #[test]
    fn from_dense_every_other_missing() {
        let dense: DenseContainer<u8> = DenseContainer::new(
            vec![0, 1, 2, 3, 4],
            vec![false, true, false, true, false],
        );
        let sparse = SparseContainer::from(dense);
        assert_eq!(sparse.len(), 5);
        assert_eq!(sparse.get(0), None);
        assert_eq!(sparse.get(1), Some(1));
        assert_eq!(sparse.get(2), None);
        assert_eq!(sparse.get(3), Some(3));
        assert_eq!(sparse.get(4), None);
    }

    //-------------------------
    #[test]
    fn into_dense_all_populated() {
        let sparse: SparseContainer<u8> = SparseContainer::from(vec![
            (0, true),
            (1, true),
            (2, true),
            (3, true),
            (4, true),
        ]);

        let dense = DenseContainer::from(sparse);
        assert_eq!(dense.len(), 5);
        assert_eq!(dense.get(0).unwrap(), 0);
        assert_eq!(dense.get(1).unwrap(), 1);
        assert_eq!(dense.get(2).unwrap(), 2);
        assert_eq!(dense.get(3).unwrap(), 3);
        assert_eq!(dense.get(4).unwrap(), 4);
    }

    #[test]
    fn into_dense_head_missing() {
        let sparse: SparseContainer<u8> = SparseContainer::from(vec![
            (0, false),
            (1, false),
            (2, true),
            (3, true),
            (4, true),
        ]);

        let dense = DenseContainer::from(sparse);
        assert_eq!(dense.len(), 5);
        assert_eq!(dense.get(0), None);
        assert_eq!(dense.get(1), None);
        assert_eq!(dense.get(2), Some(2));
        assert_eq!(dense.get(3), Some(3));
        assert_eq!(dense.get(4), Some(4));
    }

    #[test]
    fn into_dense_tail_missing() {
        let sparse: SparseContainer<u8> = SparseContainer::from(vec![
            (0, true),
            (1, true),
            (2, true),
            (3, false),
            (4, false),
        ]);
        let dense = DenseContainer::from(sparse);
        assert_eq!(dense.len(), 5);
        assert_eq!(dense.get(0), Some(0));
        assert_eq!(dense.get(1), Some(1));
        assert_eq!(dense.get(2), Some(2));
        assert_eq!(dense.get(3), None);
        assert_eq!(dense.get(4), None);
    }

    #[test]
    fn into_dense_middle_missing() {
        let sparse: SparseContainer<u8> = SparseContainer::from(vec![
            (0, true),
            (1, true),
            (2, false),
            (3, false),
            (4, true),
        ]);
        let dense = DenseContainer::from(sparse);
        assert_eq!(dense.len(), 5);
        assert_eq!(dense.get(0), Some(0));
        assert_eq!(dense.get(1), Some(1));
        assert_eq!(dense.get(2), None);
        assert_eq!(dense.get(3), None);
        assert_eq!(dense.get(4), Some(4));
    }

    #[test]
    fn into_dense_every_other_missing() {
        let sparse: SparseContainer<u8> = SparseContainer::from(vec![
            (0, false),
            (1, true),
            (2, false),
            (3, true),
            (4, false),
        ]);
        let dense = DenseContainer::from(sparse);
        assert_eq!(dense.len(), 5);
        assert_eq!(dense.get(0), None);
        assert_eq!(dense.get(1), Some(1));
        assert_eq!(dense.get(2), None);
        assert_eq!(dense.get(3), Some(3));
        assert_eq!(dense.get(4), None);
    }
    //--------------------------

    #[test]
    fn with_missing() {
        let container = sparse_container();

        assert_eq!(container.data.len(), 2);

        assert_eq!(container.data[0].0, 2);
        assert_eq!(container.data[0].1.len(), 3);
        assert!(container.data[0]
            .1
            .iter()
            .enumerate()
            .all(|(i, x)| i as f64 == x - 1.0));

        assert_eq!(container.data[1].0, 7);
        assert_eq!(container.data[1].1.len(), 4);
        assert!(container.data[1]
            .1
            .iter()
            .enumerate()
            .all(|(i, x)| i as f64 == x - 1.0));
    }

    #[test]
    fn get() {
        let container = sparse_container();

        assert_eq!(container.get(0), None);
        assert_eq!(container.get(1), None);

        assert_eq!(container.get(2), Some(1.0));
        assert_eq!(container.get(3), Some(2.0));
        assert_eq!(container.get(4), Some(3.0));

        assert_eq!(container.get(5), None);
        assert_eq!(container.get(6), None);

        assert_eq!(container.get(7), Some(1.0));
        assert_eq!(container.get(8), Some(2.0));
        assert_eq!(container.get(9), Some(3.0));
        assert_eq!(container.get(10), Some(4.0));
    }

    #[test]
    #[should_panic]
    fn get_oob_panics() {
        let container = sparse_container();
        let _x = container.get(11);
    }

    #[test]
    fn get_from_all_missing_is_none() {
        let container = SparseContainer::<u8>::from(vec![(3_u8, false); 10]);
        for i in 0..10 {
            assert_eq!(container.get(i), None);
        }
    }

    #[test]
    fn slices() {
        let container = sparse_container();
        assert_eq!(container.get_slices().iter().count(), 2);
    }

    #[test]
    fn insert_0() {
        let mut container = sparse_container();
        container.insert_overwrite(0, -1.0);

        assert_eq!(container.get(0), Some(-1.0));
        assert_eq!(container.get_slices().iter().count(), 3);
        assert_eq!(container.len(), 11);
    }

    #[test]
    fn insert_1() {
        let mut container = sparse_container();
        container.insert_overwrite(1, -1.0);

        assert_eq!(container.get(1), Some(-1.0));
        assert_eq!(container.get_slices().iter().count(), 2);
        assert_eq!(container.len(), 11);
    }

    #[test]
    fn insert_2() {
        let mut container = sparse_container();
        container.insert_overwrite(2, -1.0);

        assert_eq!(container.get(2), Some(-1.0));
        assert_eq!(container.get_slices().iter().count(), 2);
        assert_eq!(container.len(), 11);
    }

    #[test]
    fn insert_join() {
        let mut container = sparse_container();

        container.insert_overwrite(5, -1.0);
        assert_eq!(container.get(5), Some(-1.0));
        assert_eq!(container.get_slices().iter().count(), 2);
        assert_eq!(container.len(), 11);

        container.insert_overwrite(6, -1.0);
        assert_eq!(container.get(6), Some(-1.0));
        assert_eq!(container.get_slices().iter().count(), 1);
        assert_eq!(container.len(), 11);
    }

    #[test]
    fn insert_oob() {
        let mut container = sparse_container();

        container.insert_overwrite(100, -1.0);
        assert_eq!(container.len(), 101);
        assert_eq!(container.get(100), Some(-1.0));
        assert_eq!(container.get_slices().iter().count(), 3);
    }

    #[test]
    fn remove_0() {
        let mut container = sparse_container();
        let x = container.remove(0);
        assert_eq!(x, None);
    }

    #[test]
    fn remove_2() {
        let mut container = sparse_container();
        let x = container.remove(2);

        assert_eq!(x, Some(1.0));
        assert_eq!(container.data[0].0, 3);

        assert_eq!(container.data[0].1.len(), 2);

        assert_eq!(container.data[0].1[0], 2.0);
        assert_eq!(container.data[0].1[1], 3.0);

        assert_eq!(container.data.len(), 2);
    }

    #[test]
    fn remove_3() {
        let mut container = sparse_container();
        let x = container.remove(3);

        assert_eq!(x, Some(2.0));

        // should have split the first slice
        assert_eq!(container.data[0].0, 2);
        assert_eq!(container.data[0].1.len(), 1);
        assert_eq!(container.data[0].1[0], 1.0);

        assert_eq!(container.data[1].0, 4);
        assert_eq!(container.data[1].1.len(), 1);
        assert_eq!(container.data[1].1[0], 3.0);

        assert_eq!(container.data.len(), 3);
    }

    #[test]
    fn remove_4() {
        let mut container = sparse_container();
        let x = container.remove(4);

        assert_eq!(x, Some(3.0));

        // should have split the first slice
        assert_eq!(container.data[0].0, 2);
        assert_eq!(container.data[0].1.len(), 2);
        assert_eq!(container.data[0].1[0], 1.0);
        assert_eq!(container.data[0].1[1], 2.0);

        assert_eq!(container.data.len(), 2);
    }

    #[test]
    fn push_start_missing() {
        let mut container: SparseContainer<u8> = SparseContainer::default();

        assert!(container.is_empty());
        assert_eq!(container.len(), 0);

        container.push(None);

        assert!(!container.is_empty());
        assert_eq!(container.len(), 1);

        container.push(None);

        assert!(!container.is_empty());
        assert_eq!(container.len(), 2);

        container.push(Some(1));

        assert!(!container.is_empty());
        assert_eq!(container.len(), 3);

        assert_eq!(container.get(0), None);
        assert_eq!(container.get(1), None);
        assert_eq!(container.get(2), Some(1));
    }

    #[test]
    fn push_to_from_vec_ctor() {
        let empty_vec: Vec<u8> = Vec::new();
        let mut container: SparseContainer<u8> =
            SparseContainer::from(empty_vec);

        container.push(None);
        container.push(Some(1));
        container.push(Some(2));
        container.push(None);
        container.push(Some(4));

        assert_eq!(container.len(), 5);

        assert_eq!(container.get(0), None);
        assert_eq!(container.get(1), Some(1));
        assert_eq!(container.get(2), Some(2));
        assert_eq!(container.get(3), None);
        assert_eq!(container.get(4), Some(4));
    }
}
