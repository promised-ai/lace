use cc::assignment::Assignment;
use std::ops::{Index, IndexMut};

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct DataContainer<T>
where
    T: Clone,
{
    pub data: Vec<T>,
    pub present: Vec<bool>,
}

// For pulling data from features for saving
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum FeatureData {
    Continuous(DataContainer<f64>),
    Categorical(DataContainer<u8>),
}

impl<T> DataContainer<T>
where
    T: Clone,
{
    pub fn new(data: Vec<T>) -> DataContainer<T> {
        let n = data.len();
        DataContainer {
            data: data,
            present: vec![true; n],
        }
    }

    pub fn empty() -> DataContainer<T> {
        DataContainer {
            data: vec![],
            present: vec![],
        }
    }

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
    pub fn group_by<'a>(&self, asgn: &'a Assignment) -> Vec<Vec<T>> {
        assert!(asgn.validate().is_valid());
        assert_eq!(asgn.len(), self.len());
        // FIXME: Filter on `present` using better zip library
        (0..asgn.ncats)
            .map(|k| {
                let grp: Vec<T> = self.data
                    .iter()
                    .zip(asgn.asgn.iter())
                    .filter(|&(_, z)| *z == k)
                    .map(|(x, _)| x.clone())
                    .collect();
                assert!(!grp.is_empty());
                grp
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
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
}
