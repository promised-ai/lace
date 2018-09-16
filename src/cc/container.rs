use cc::assignment::Assignment;
use std::ops::{Index, IndexMut};

/// Stores present or missing data
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct DataContainer<T>
where
    T: Clone,
{
    pub data: Vec<T>,
    pub present: Vec<bool>,
}

/// Used when pulling data from features for saving
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum FeatureData {
    /// Univariate continuous data
    Continuous(DataContainer<f64>),
    /// Categorical data
    Categorical(DataContainer<u8>),
}

impl<T> DataContainer<T>
where
    T: Clone,
{
    /// New container with all present data
    pub fn new(data: Vec<T>) -> DataContainer<T> {
        let n = data.len();
        DataContainer {
            data: data,
            present: vec![true; n],
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
    /// - dummy_val: The value with which to replace missing values. This
    ///   should be in the support of the `Feature`'s distribution. For
    ///   example, if the column is categorical, then then `dummy_val` should
    ///   be in 0, ..., k-1
    /// - `pred`: A function, `pred(data[i])` that returns `true` if the data[i]
    ///   is present.
    pub fn with_filter<F>(
        mut data: Vec<T>,
        dummy_val: T,
        pred: F,
    ) -> DataContainer<T>
    where
        F: Fn(&T) -> bool,
    {
        let n = data.len();
        let mut present: Vec<bool> = vec![true; n];
        for i in 0..n {
            if !pred(&data[i]) {
                present[i] = false;
                data[i] = dummy_val.clone();
            }
        }
        DataContainer {
            data: data,
            present: present,
        }
    }

    /// Push a new potential value to the container.
    ///
    /// If `val` is `None`, then a missing datum with placeholder, `dummy_val`,
    /// is inserted.
    pub fn push(&mut self, val: Option<T>, dummy_val: T) {
        match val {
            Some(x) => {
                self.data.push(x);
                self.present.push(true);
            }
            None => {
                self.data.push(dummy_val);
                self.present.push(false);
            }
        }
    }

    // TODO: Add method to construct sufficient statistics instead of
    // retuning data
    // XXX: might be faster to use nested for loop?
    /// Group the present data according an `Assignment`
    ///
    /// # Example
    ///
    /// ```rust
    /// # extern crate rand;
    /// # extern crate braid;
    /// # use braid::cc::AssignmentBuilder;
    /// # use braid::cc::DataContainer;
    /// let mut rng = rand::thread_rng();
    /// let assignment = AssignmentBuilder::from_vec(vec![0, 0, 2, 1])
    ///     .build(&mut rng);
    ///
    /// let container = DataContainer::new(vec![1.0, 2.0, 3.0, 4.0]);
    /// let xs_grouped = container.group_by(&assignment);
    ///
    /// assert_eq!(xs_grouped.len(), 3);
    /// assert_eq!(xs_grouped[0], vec![1.0, 2.0]);
    /// assert_eq!(xs_grouped[1], vec![4.0]);
    /// assert_eq!(xs_grouped[2], vec![3.0]);
    /// ```
    pub fn group_by<'a>(&self, asgn: &'a Assignment) -> Vec<Vec<T>> {
        // assert!(asgn.validate().is_valid());
        assert_eq!(asgn.len(), self.len());
        // TODO: Filter on `present` using better zip library
        (0..asgn.ncats)
            .map(|k| {
                let grp: Vec<T> = self
                    .data
                    .iter()
                    .zip(self.present.iter())
                    .zip(asgn.asgn.iter())
                    .filter(|&((_, r), z)| *z == k && *r)
                    .map(|((x, _), _)| x.clone())
                    .collect();
                grp
            }).collect()
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
    fn index_mut<'a>(&'a mut self, ix: usize) -> &'a mut T {
        &mut self.data[ix]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rv;
    use self::rv::dist::Gamma;
    use std::f64::NAN;

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
    fn default_container_bool_should_all_construct_properly() {
        let data: Vec<bool> = vec![true, false, false, true];
        let container = DataContainer::new(data);

        assert_eq!(container.data.len(), 4);
        assert_eq!(container.present.len(), 4);

        assert!(container.present.iter().all(|&x| x));

        assert_eq!(container.data[0], true);
        assert_eq!(container.data[1], false);
        assert_eq!(container.data[2], false);
        assert_eq!(container.data[3], true);
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
        let container = DataContainer::with_filter(data, 0, |&x| x != 99);

        assert!(container.present[0]);
        assert!(container.present[1]);
        assert!(!container.present[2]);
        assert!(container.present[3]);

        assert_eq!(container[0], 0);
        assert_eq!(container[1], 1);
        assert_eq!(container[2], 0);
        assert_eq!(container[3], 3);
    }

    #[test]
    fn filter_container_f64_nan_should_tag_and_set_missing_values() {
        let data: Vec<f64> = vec![0.0, 1.0, NAN, 3.0];

        // the filter identifies present (non-missing) values
        let container =
            DataContainer::with_filter(data, 0.0, |&x| x.is_finite());

        assert!(container.present[0]);
        assert!(container.present[1]);
        assert!(!container.present[2]);
        assert!(container.present[3]);

        assert_relative_eq!(container[0], 0.0, epsilon = 1E-10);
        assert_relative_eq!(container[1], 1.0, epsilon = 1E-10);
        assert_relative_eq!(container[2], 0.0, epsilon = 1E-10);
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
    fn group_by() {
        let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6];
        let asgn = Assignment {
            alpha: 1.0,
            asgn: vec![0, 0, 1, 1, 0, 0, 2],
            counts: vec![4, 2, 1],
            ncats: 3,
            prior: Gamma::new(1.0, 1.0).unwrap(),
        };
        let container = DataContainer::new(data);
        let xs = container.group_by(&asgn);
        assert_eq!(xs.len(), 3);

        assert_eq!(xs[0].len(), 4);
        assert_eq!(xs[1].len(), 2);
        assert_eq!(xs[2].len(), 1);

        assert_eq!(xs[0][0], 0);
        assert_eq!(xs[0][1], 1);
        assert_eq!(xs[1][0], 2);
        assert_eq!(xs[1][1], 3);
        assert_eq!(xs[0][2], 4);
        assert_eq!(xs[0][3], 5);
        assert_eq!(xs[2][0], 6);
    }

    #[test]
    fn group_by_with_missing_should_not_return_missing_no_empty_bins() {
        let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6];
        let asgn = Assignment {
            alpha: 1.0,
            asgn: vec![0, 0, 1, 1, 0, 0, 2],
            counts: vec![4, 2, 1],
            ncats: 3,
            prior: Gamma::new(1.0, 1.0).unwrap(),
        };
        let container = DataContainer {
            data: data,
            present: vec![true, true, false, true, false, true, true],
        };
        let xs = container.group_by(&asgn);

        assert_eq!(xs.len(), 3);
        assert_eq!(xs[0].len(), 3);
        assert_eq!(xs[1].len(), 1);
        assert_eq!(xs[2].len(), 1);

        assert_eq!(xs[0], vec![0, 1, 5]);
        assert_eq!(xs[1], vec![3]);
        assert_eq!(xs[2], vec![6]);
    }

    #[test]
    fn group_by_with_missing_should_not_return_missing_empty_bin() {
        let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6];
        let asgn = Assignment {
            alpha: 1.0,
            asgn: vec![0, 0, 1, 1, 0, 0, 2],
            counts: vec![4, 2, 1],
            ncats: 3,
            prior: Gamma::new(1.0, 1.0).unwrap(),
        };
        let container = DataContainer {
            data: data,
            present: vec![true, true, false, true, false, true, false],
        };
        let xs = container.group_by(&asgn);

        assert_eq!(xs.len(), 3);
        assert_eq!(xs[0].len(), 3);
        assert_eq!(xs[1].len(), 1);
        assert!(xs[2].is_empty());

        assert_eq!(xs[0], vec![0, 1, 5]);
        assert_eq!(xs[1], vec![3]);
    }

    #[test]
    fn group_by_should_ignore_unassigned_entries() {
        let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6];
        let mut asgn = Assignment {
            alpha: 1.0,
            asgn: vec![0, 0, 1, 1, 0, 0, 1],
            counts: vec![4, 3],
            ncats: 2,
            prior: Gamma::new(1.0, 1.0).unwrap(),
        };

        asgn.unassign(3);

        let container = DataContainer::new(data);
        let xs = container.group_by(&asgn);

        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0].len(), 4);
        assert_eq!(xs[1].len(), 2);

        assert_eq!(xs[0], vec![0, 1, 4, 5]);
        assert_eq!(xs[1], vec![2, 6]);
    }
}
