/// A data container
pub trait Container<T: Copy> {
    /// get the data slices and the start indices
    fn get_slices(&self) -> Vec<(usize, &[T])>;

    /// Get the entry at ix if it exists
    fn get(&self, ix: usize) -> Option<T>;

    /// Insert or overwrite an entry at ix
    fn insert_overwrite(&mut self, ix: usize, x: T);

    /// Remove and return the entry at ix if it exists. Used to mark a present
    /// datum as missing, not to completely remove a record.
    fn remove(&mut self, ix: usize) -> Option<T>;
}

/// The baseline data structure
pub struct DenseContainer<T: Copy + Default> {
    /// The actual values of the data. Uses `Default::default()`
    values: Vec<T>,
    /// Tells whether each values is present or missing
    present: Vec<bool>,
}

impl<T: Copy + Default> DenseContainer<T> {
    pub fn new(values: Vec<T>, present: Vec<bool>) -> DenseContainer<T> {
        DenseContainer { values, present }
    }
}

impl<T: Copy + Default> Container<T> for DenseContainer<T> {
    fn get_slices(&self) -> Vec<(usize, &[T])> {
        vec![(0, self.values.as_slice())]
    }

    fn get(&self, ix: usize) -> Option<T> {
        if self.present[ix] {
            Some(self.values[ix])
        } else {
            None
        }
    }

    fn insert_overwrite(&mut self, ix: usize, x: T) {
        self.values[ix] = x;
        self.present[ix] = true;
    }

    fn remove(&mut self, ix: usize) -> Option<T> {
        if self.present[ix] {
            let out = self.values[ix];
            self.values[ix] = T::default();
            self.present[ix] = false;
            Some(out)
        } else {
            None
        }
    }
}

/// A sparse container stores contiguous vertical slices of data
pub struct VecContainer<T: Copy> {
    /// Each entry is the index of the index of the first entry
    data: Vec<(usize, Vec<T>)>,
}

impl<T: Copy> VecContainer<T> {
    pub fn new(mut xs: Vec<T>, present: &Vec<bool>) -> VecContainer<T> {
        let mut data: Vec<(usize, Vec<T>)> = Vec::new();
        let mut filling: bool = false;

        for (i, (&pr, x)) in present.iter().zip(xs.drain(..)).enumerate() {
            if filling {
                if pr {
                    // push to last data vec
                    data.last_mut().unwrap().1.push(x);
                } else {
                    // stop filling
                    filling = false;
                }
            } else if pr {
                // create a new data vec and startfilling
                data.push((i, vec![x]));
                filling = true;
            }
        }

        VecContainer { data }
    }

    /// Determines whether an insert joined two data slices and merges them
    /// internally if so.
    fn check_merge_next(&mut self, ix: usize) {
        if ix < self.data.len() {
            let start_ix = self.data[ix].0 + self.data[ix].1.len();
            if start_ix == self.data[ix + 1].0 {
                let (_, mut bottom) = self.data.remove(ix + 1);
                self.data[ix].1.append(&mut bottom);
            }
        }
    }
}

impl<T: Copy> Container<T> for VecContainer<T> {
    fn get_slices(&self) -> Vec<(usize, &[T])> {
        self.data
            .iter()
            .map(|(ix, xs)| (*ix, xs.as_slice()))
            .collect()
    }

    fn get(&self, ix: usize) -> Option<T> {
        if self.data[0].0 > ix {
            None
        } else {
            let result = self.data.binary_search_by(|entry| entry.0.cmp(&ix));

            match result {
                Ok(index) => Some(self.data[index].1[0]),
                Err(index) => {
                    let n = self.data[index - 1].1.len();
                    let start_ix = self.data[index - 1].0;
                    let local_ix = ix - start_ix;

                    if ix >= start_ix + n {
                        // Between data slices. Missing data.
                        None
                    } else {
                        Some(self.data[index - 1].1[local_ix])
                    }
                }
            }
        }
    }

    fn insert_overwrite(&mut self, ix: usize, x: T) {
        let result = self.data.binary_search_by(|entry| entry.0.cmp(&ix));
        match result {
            Ok(index) => {
                self.data[index].1[0] = x;
            }

            Err(index) => {
                if index == 0 {
                    // data is missing and before any existing data
                    if ix == self.data[0].0 - 1 {
                        // inserted datum sits on top of the top slice
                        // TODO: need to check if linked two slice
                        self.data[0].0 = ix;
                        self.data[0].1.insert(0, x);
                    } else {
                        // inserted data is not contiguous with top data
                        self.data.insert(0, (ix, vec![x]));
                        // TODO: might be better to do manually here
                        self.check_merge_next(0);
                    }
                } else {
                    let n = self.data[index - 1].1.len();
                    let start_ix = self.data[index - 1].0;
                    let end_ix = start_ix + n;

                    // check if present
                    let present = ix < end_ix;

                    if present {
                        let local_ix = ix - start_ix;
                        self.data[index - 1].1[local_ix] = x;
                    } else {
                        if ix == end_ix {
                            self.data[index - 1].1.push(x);
                            self.check_merge_next(index - 1);
                        } else {
                            self.data.insert(index, (ix, vec![x]));
                        }
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

/// A container that holds the present data in one contiguous vector and the
/// intervals in another.
pub struct LookupContainer<T: Copy> {
    /// (user-facing index, internal index, length)
    lookup: Vec<(usize, usize, usize)>,
    /// The data
    data: Vec<T>,
}

impl<T: Copy> LookupContainer<T> {
    pub fn new(xs: &Vec<T>, present: &Vec<bool>) -> LookupContainer<T> {
        let mut data: Vec<T> = Vec::new();
        let mut lookup: Vec<(usize, usize, usize)> = Vec::new();

        let mut filling: bool = false;
        let mut n: usize = 0;

        for (i, (&pr, &x)) in present.iter().zip(xs.iter()).enumerate() {
            if filling {
                if pr {
                    // push to last data vec
                    n += 1;
                    data.push(x);
                } else {
                    // stop filling
                    lookup.last_mut().unwrap().2 = n;
                    filling = false;
                }
            } else if pr {
                n = 1;
                lookup.push((i, data.len(), 1));
                data.push(x);
                filling = true;
            }
        }

        if filling {
            lookup.last_mut().unwrap().2 = n;
        }

        LookupContainer { data, lookup }
    }

    /// Increments the value index of lookups after index `ix` as a result of
    /// inserting a new datum.
    fn incr_indices_after_ix(&mut self, ix: usize) {
        self.lookup.iter_mut().skip(ix).for_each(|lkp| lkp.1 += 1);
    }

    /// Determines whether an insert joined two data slices and merges them
    /// internally if so.
    fn check_merge_next(&mut self, ix: usize) {
        if ix < self.lookup.len() {
            let start_ix = self.lookup[ix].0 + self.lookup[ix].2;
            if start_ix == self.lookup[ix + 1].0 {
                let lkp = self.lookup.remove(ix + 1);
                self.lookup[ix].2 += lkp.2;
            }
        }
    }
}

impl<T: Copy> Container<T> for LookupContainer<T> {
    fn get_slices(&self) -> Vec<(usize, &[T])> {
        self.lookup
            .iter()
            .map(|&(ix, iix, n)| {
                let xs = unsafe {
                    let ptr = self.data.as_ptr().add(iix);
                    std::slice::from_raw_parts(ptr, n)
                };
                (ix, xs)
            })
            .collect()
    }

    fn get(&self, ix: usize) -> Option<T> {
        if ix < self.lookup[0].0 {
            None
        } else {
            let target = self.lookup.iter().find(|(i, _, n)| *i + n > ix);

            match target {
                Some((i, _, _)) if *i > ix => None,
                Some((i, iix, _)) => Some(self.data[iix + ix - i]),
                None => None,
            }
        }
    }

    fn insert_overwrite(&mut self, ix: usize, x: T) {
        let result = self.lookup.binary_search_by(|entry| entry.0.cmp(&ix));
        match result {
            Ok(index) => {
                let value_ix = self.lookup[index].1;
                self.data[value_ix] = x;
            }
            Err(index) => {
                if index == 0 {
                    // data is missing and before any existing data
                    self.data.insert(0, x);
                    if ix == self.lookup[0].0 - 1 {
                        // inserted datum sits on top of the top slice
                        self.lookup[0].0 = ix;
                        self.lookup[0].2 += 1;
                    } else {
                        // inserted data is not contiguous with top data
                        self.lookup.insert(0, (ix, 0, 1));
                    }
                    self.incr_indices_after_ix(1);
                } else if index == self.lookup.len() {
                    self.lookup.push((ix, self.data.len(), 1));
                    self.data.push(x);
                } else {
                    let (start_ix, value_ix, n) = self.lookup[index - 1];
                    let end_ix = start_ix + n;

                    // check if present
                    let present = ix < end_ix;
                    let local_ix = value_ix + ix - start_ix;

                    if present {
                        self.data[local_ix] = x;
                    } else {
                        let local_ix = value_ix + ix - start_ix;
                        self.data.insert(local_ix, x);
                        if ix == end_ix {
                            self.lookup[index - 1].2 += 1;
                            self.incr_indices_after_ix(index);
                            self.check_merge_next(index - 1);
                        } else {
                            self.lookup.insert(index, (ix, local_ix, 1));
                            self.incr_indices_after_ix(index + 1);
                        }
                    }
                }
            }
        }
    }

    fn remove(&mut self, ix: usize) -> Option<T> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod lookup_container {
        use super::*;

        fn lookup_container() -> LookupContainer<f64> {
            let xs: Vec<f64> =
                vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
            let present = vec![
                false, false, true, true, true, false, false, true, true, true,
                true,
            ];
            LookupContainer::new(&xs, &present)
        }

        #[test]
        fn ctor() {
            let container = lookup_container();
            assert_eq!(container.data.len(), 7);
            assert_eq!(container.lookup.len(), 2);
            assert_eq!(container.lookup[0], (2, 0, 3));
            assert_eq!(container.lookup[1], (7, 3, 4));
        }

        #[test]
        fn get() {
            let container = lookup_container();

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
        fn get_oob_is_none() {
            let container = lookup_container();

            assert_eq!(container.get(11), None);
            assert_eq!(container.get(2000), None);
        }

        #[test]
        fn insert_0() {
            let mut container = lookup_container();
            container.insert_overwrite(0, -1.0);

            assert_eq!(container.get(0), Some(-1.0));
            assert_eq!(container.lookup.len(), 3);
            assert_eq!(container.get_slices().iter().count(), 3);
            assert_eq!(container.lookup[0], (0, 0, 1));
            assert_eq!(container.lookup[1], (2, 1, 3));
            assert_eq!(container.lookup[2], (7, 4, 4));
        }

        #[test]
        fn insert_1() {
            let mut container = lookup_container();
            container.insert_overwrite(1, -1.0);

            assert_eq!(container.get(1), Some(-1.0));
            assert_eq!(container.lookup.len(), 2);
            assert_eq!(container.get_slices().iter().count(), 2);
            assert_eq!(container.lookup[0], (1, 0, 4));
            assert_eq!(container.lookup[1], (7, 4, 4));
        }

        #[test]
        fn insert_2() {
            let mut container = lookup_container();
            container.insert_overwrite(2, -1.0);

            assert_eq!(container.get(2), Some(-1.0));
            assert_eq!(container.lookup.len(), 2);
            assert_eq!(container.get_slices().iter().count(), 2);
            assert_eq!(container.lookup[0], (2, 0, 3));
            assert_eq!(container.lookup[1], (7, 3, 4));
        }

        #[test]
        fn insert_5() {
            let mut container = lookup_container();
            container.insert_overwrite(5, -1.0);

            assert_eq!(container.get(5), Some(-1.0));
            assert_eq!(container.get(6), None);
            assert_eq!(container.lookup.len(), 2);
            assert_eq!(container.get_slices().iter().count(), 2);
            assert_eq!(container.lookup[0], (2, 0, 4));
            assert_eq!(container.lookup[1], (7, 4, 4));
        }

        #[test]
        fn insert_to_join() {
            let mut container = lookup_container();
            container.insert_overwrite(5, -1.0);
            container.insert_overwrite(6, -2.0);

            assert_eq!(container.get(5), Some(-1.0));
            assert_eq!(container.get(6), Some(-2.0));
            assert_eq!(container.lookup.len(), 1);
            assert_eq!(container.lookup[0], (2, 0, 9));
        }

        #[test]
        fn insert_oob() {
            let mut container = lookup_container();
            container.insert_overwrite(100, -1.0);

            assert_eq!(container.get(100), Some(-1.0));
            assert_eq!(container.lookup.len(), 3);
            assert_eq!(container.lookup[2], (100, 7, 1));
            assert_eq!(container.get_slices().iter().count(), 3);
        }
    }

    mod vec_container {
        use super::*;

        fn vec_container() -> VecContainer<f64> {
            let xs: Vec<f64> =
                vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
            let present = vec![
                false, false, true, true, true, false, false, true, true, true,
                true,
            ];
            VecContainer::new(xs, &present)
        }

        #[test]
        fn vec_container_ctor() {
            let container = vec_container();

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
        fn vec_container_get() {
            let container = vec_container();

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
        fn vec_container_get_oob_is_none() {
            let container = vec_container();

            assert_eq!(container.get(11), None);
            assert_eq!(container.get(2000), None);
        }

        #[test]
        fn vec_container_slices() {
            let container = vec_container();
            assert_eq!(container.get_slices().iter().count(), 2);
        }

        #[test]
        fn vec_insert_0() {
            let mut container = vec_container();
            container.insert_overwrite(0, -1.0);

            assert_eq!(container.get(0), Some(-1.0));
            assert_eq!(container.get_slices().iter().count(), 3);
        }

        #[test]
        fn vec_insert_1() {
            let mut container = vec_container();
            container.insert_overwrite(1, -1.0);

            assert_eq!(container.get(1), Some(-1.0));
            assert_eq!(container.get_slices().iter().count(), 2);
        }

        #[test]
        fn vec_insert_2() {
            let mut container = vec_container();
            container.insert_overwrite(2, -1.0);

            assert_eq!(container.get(2), Some(-1.0));
            assert_eq!(container.get_slices().iter().count(), 2);
        }

        #[test]
        fn vec_insert_join() {
            let mut container = vec_container();

            container.insert_overwrite(5, -1.0);
            assert_eq!(container.get(5), Some(-1.0));
            assert_eq!(container.get_slices().iter().count(), 2);

            container.insert_overwrite(6, -1.0);
            assert_eq!(container.get(6), Some(-1.0));
            assert_eq!(container.get_slices().iter().count(), 1);
        }

        #[test]
        fn insert_oob() {
            let mut container = vec_container();

            container.insert_overwrite(100, -1.0);
            assert_eq!(container.get(100), Some(-1.0));
            assert_eq!(container.get_slices().iter().count(), 3);
        }

        #[test]
        fn vec_remove_0() {
            let mut container = vec_container();
            let x = container.remove(0);
            assert_eq!(x, None);
        }

        #[test]
        fn vec_remove_2() {
            let mut container = vec_container();
            let x = container.remove(2);

            assert_eq!(x, Some(1.0));
            assert_eq!(container.data[0].0, 3);

            assert_eq!(container.data[0].1.len(), 2);

            assert_eq!(container.data[0].1[0], 2.0);
            assert_eq!(container.data[0].1[1], 3.0);

            assert_eq!(container.data.len(), 2);
        }

        #[test]
        fn vec_remove_3() {
            let mut container = vec_container();
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
        fn vec_remove_4() {
            let mut container = vec_container();
            let x = container.remove(4);

            assert_eq!(x, Some(3.0));

            // should have split the first slice
            assert_eq!(container.data[0].0, 2);
            assert_eq!(container.data[0].1.len(), 2);
            assert_eq!(container.data[0].1[0], 1.0);
            assert_eq!(container.data[0].1[1], 2.0);

            assert_eq!(container.data.len(), 2);
        }
    }
}
