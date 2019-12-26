use std::convert::TryFrom;
use std::ops::{Index, IndexMut};

use braid_stats::Datum;
use serde::{Deserialize, Serialize};

/// Stores present or missing data
#[derive(Serialize, Deserialize, Debug, PartialEq, PartialOrd, Clone)]
pub struct DataContainer<T>
where
    T: Clone,
{
    /// The data
    pub data: Vec<T>,
    /// An indicator for each datum. `present[i]` is `true` if datum `i` is not
    /// missing.
    pub present: Vec<bool>,
}

impl<T> DataContainer<T>
where
    T: Clone + TryFrom<Datum> + Default,
{
    /// New container with all present data
    pub fn new(data: Vec<T>) -> DataContainer<T> {
        let n = data.len();
        DataContainer {
            data,
            present: vec![true; n],
        }
    }

    /// New container with all missing data
    pub fn all_missing(n: usize) -> DataContainer<T> {
        DataContainer {
            data: vec![T::default(); n],
            present: vec![false; n],
        }
    }

    /// Initialize and empty container
    pub fn empty() -> DataContainer<T> {
        DataContainer {
            data: vec![],
            present: vec![],
        }
    }

    /// Initialize data that is present if the predicate function `pred`
    /// returns `true`.
    ///
    /// # Arguments
    ///
    /// - data: A vector of values
    /// - `pred`: A function, `pred(data[i])` that returns `true` if `data[i]`
    ///   is present.
    pub fn with_filter<F>(mut data: Vec<T>, pred: F) -> DataContainer<T>
    where
        F: Fn(&T) -> bool,
    {
        let n = data.len();
        let mut present: Vec<bool> = vec![true; n];
        for i in 0..n {
            if !pred(&data[i]) {
                present[i] = false;
                data[i] = T::default();
            }
        }
        DataContainer { data, present }
    }

    /// Push a new potential value to the container.
    ///
    /// If `val` is `None`, then a missing datum with placeholder, T::default()
    /// is inserted.
    pub fn push(&mut self, val: Option<T>) {
        match val {
            Some(x) => {
                self.data.push(x);
                self.present.push(true);
            }
            None => {
                self.data.push(T::default());
                self.present.push(false);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn zip(&self) -> impl Iterator<Item = (&T, &bool)> {
        self.data.iter().zip(self.present.iter())
    }

    pub fn push_datum(&mut self, x: Datum) {
        match x {
            Datum::Missing => self.push(None),
            _ => {
                if let Ok(val) = T::try_from(x) {
                    self.push(Some(val));
                } else {
                    panic!("failed to convert datum");
                }
            }
        }
    }

    pub fn insert_datum(&mut self, row_ix: usize, x: Datum) {
        match x {
            Datum::Missing => {
                self.present[row_ix] = false;
                self.data[row_ix] = T::default();
            }
            _ => {
                if let Ok(val) = T::try_from(x) {
                    self.present[row_ix] = true;
                    self.data[row_ix] = val;
                } else {
                    panic!("failed to convert datum");
                }
            }
        }
    }
}

impl<T> Default for DataContainer<T>
where
    T: Clone,
{
    fn default() -> Self {
        DataContainer {
            data: vec![],
            present: vec![],
        }
    }
}

impl<T> Index<usize> for DataContainer<T>
where
    T: Clone,
{
    type Output = T;
    fn index(&self, ix: usize) -> &T {
        &self.data[ix]
    }
}

impl<T> IndexMut<usize> for DataContainer<T>
where
    T: Clone,
{
    fn index_mut(&mut self, ix: usize) -> &mut T {
        &mut self.data[ix]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use std::f64::NAN;

    #[test]
    fn all_missing_should_have_all_missing_data() {
        let container: DataContainer<u8> = DataContainer::all_missing(31);
        assert_eq!(container.len(), 31);
        assert_eq!(container.present.len(), 31);
        assert_eq!(container.data.len(), 31);
        for ix in 0..31 {
            assert!(!container.present[ix]);
            assert_eq!(container.data[ix], 0);
        }
    }

    #[test]
    fn default_container_f64_should_all_construct_properly() {
        let data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let container = DataContainer::new(data);

        assert_eq!(container.data.len(), 4);
        assert_eq!(container.present.len(), 4);

        assert!(container.present.iter().all(|&x| x));

        assert_relative_eq!(container.data[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(container.data[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(container.data[2], 2.0, epsilon = 1e-10);
        assert_relative_eq!(container.data[3], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn default_container_u8_should_all_construct_properly() {
        let data: Vec<u8> = vec![0, 1, 2, 3];
        let container = DataContainer::new(data);

        assert_eq!(container.data.len(), 4);
        assert_eq!(container.present.len(), 4);

        assert!(container.present.iter().all(|&x| x));

        assert_eq!(container.data[0], 0);
        assert_eq!(container.data[1], 1);
        assert_eq!(container.data[2], 2);
        assert_eq!(container.data[3], 3);
    }

    #[test]
    fn test_index_impl() {
        let data: Vec<u8> = vec![0, 1, 2, 3];
        let container = DataContainer::new(data);

        assert_eq!(container[0], 0);
        assert_eq!(container[1], 1);
        assert_eq!(container[2], 2);
        assert_eq!(container[3], 3);
    }

    #[test]
    fn test_index_mut_impl() {
        let data: Vec<u8> = vec![0, 1, 2, 3];
        let mut container = DataContainer::new(data);

        assert_eq!(container[0], 0);
        assert_eq!(container[1], 1);
        assert_eq!(container[2], 2);
        assert_eq!(container[3], 3);

        container[2] = 97;

        assert_eq!(container[0], 0);
        assert_eq!(container[1], 1);
        assert_eq!(container[2], 97);
        assert_eq!(container[3], 3);
    }

    #[test]
    fn filter_container_u8_should_tag_and_set_missing_values() {
        let data: Vec<u8> = vec![0, 1, 99, 3];

        // the filter identifies present (non-missing) values
        let container = DataContainer::with_filter(data, |&x| x != 99);

        assert!(container.present[0]);
        assert!(container.present[1]);
        assert!(!container.present[2]);
        assert!(container.present[3]);

        assert_eq!(container[0], 0);
        assert_eq!(container[1], 1);
        assert_eq!(container[2], u8::default());
        assert_eq!(container[3], 3);
    }

    #[test]
    fn filter_container_f64_nan_should_tag_and_set_missing_values() {
        let data: Vec<f64> = vec![0.0, 1.0, NAN, 3.0];

        // the filter identifies present (non-missing) values
        let container = DataContainer::with_filter(data, |&x| x.is_finite());

        assert!(container.present[0]);
        assert!(container.present[1]);
        assert!(!container.present[2]);
        assert!(container.present[3]);

        assert_relative_eq!(container[0], 0.0, epsilon = 1E-10);
        assert_relative_eq!(container[1], 1.0, epsilon = 1E-10);
        assert_relative_eq!(container[2], f64::default(), epsilon = 1E-10);
        assert_relative_eq!(container[3], 3.0, epsilon = 1E-10);
    }

    #[test]
    fn set_value() {
        let data: Vec<u8> = vec![0, 1, 2, 3];
        let mut container = DataContainer::new(data);

        assert_eq!(container[0], 0);
        assert_eq!(container[1], 1);
        assert_eq!(container[2], 2);
        assert_eq!(container[3], 3);

        container[2] = 22;

        assert_eq!(container[0], 0);
        assert_eq!(container[1], 1);
        assert_eq!(container[2], 22);
        assert_eq!(container[3], 3);
    }

    #[test]
    fn append_datum_u8_present() {
        let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6];
        let mut container = DataContainer::new(data);
        let x = Datum::Categorical(12);

        container.push_datum(x);
        assert_eq!(container[7], 12);
    }

    #[test]
    fn append_datum_u8_missing() {
        let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6];
        let mut container = DataContainer::new(data);
        let x = Datum::Missing;

        container.push_datum(x);
        assert_eq!(container[7], u8::default());
    }

    #[test]
    fn append_datum_f64_present() {
        let data: Vec<f64> = vec![1.1, 2.2, 3.3];
        let mut container = DataContainer::new(data);
        let x = Datum::Continuous(4.4);

        container.push_datum(x);
        assert_relative_eq!(container[3], 4.4, epsilon = 1E-8);
    }

    #[test]
    fn append_datum_f64_missing() {
        let data: Vec<f64> = vec![1.1, 2.2, 3.3];
        let mut container = DataContainer::new(data);
        let x = Datum::Missing;

        container.push_datum(x);
        assert_relative_eq!(container[3], f64::default(), epsilon = 1E-8);
    }
}
