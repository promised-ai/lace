use std::collections::{BTreeMap, HashSet};
use std::hash::Hash;

pub trait UniqueCollection {
    type Item;

    /// Get the unique values in the collection
    ///
    /// # Example
    ///
    /// ```rust
    /// # use braid_utils::unique::UniqueCollection;
    ///
    /// let xs: Vec<u8> = vec![0, 1, 2, 1, 0, 2, 1, 0, 4];
    ///
    /// let values = xs.unique_values();
    ///
    /// assert_eq!(values.len(), 4);
    /// assert_eq!(values.len(), xs.n_unique());
    ///
    /// assert!(xs.contains(&0));
    /// assert!(xs.contains(&1));
    /// assert!(xs.contains(&2));
    /// assert!(xs.contains(&4));
    /// ```
    fn unique_values(&self) -> Vec<Self::Item>;

    /// Get the number of unique values
    ///
    /// # Example
    ///
    /// ```rust
    /// # use braid_utils::unique::UniqueCollection;
    ///
    /// let xs: Vec<u8> = vec![0, 1, 2, 1, 0, 2, 1, 0, 4];
    ///
    /// assert_eq!(xs.n_unique(), 4);
    /// ```
    fn n_unique(&self) -> usize {
        self.unique_values().len()
    }

    /// Get minimum of the number of unique values and the cutoff
    ///
    /// # Example
    ///
    /// ```rust
    /// # use braid_utils::unique::UniqueCollection;
    ///
    /// let xs: Vec<u8> = vec![0, 1, 2, 1, 0, 2, 1, 0, 4];
    ///
    /// assert_eq!(xs.n_unique_cutoff(4), 4);
    /// assert_eq!(xs.n_unique_cutoff(10), 4);
    /// assert_eq!(xs.n_unique_cutoff(3), 3);
    /// ```
    fn n_unique_cutoff(&self, cutoff: usize) -> usize;

    /// Get the unique values indexed by a `usize` ID
    ///
    /// # Example
    ///
    /// ```rust
    /// # use braid_utils::unique::UniqueCollection;
    ///
    /// let xs: Vec<u8> = vec![0, 1, 2, 1, 0, 2, 1, 0, 4];
    ///
    /// let map = xs.value_map();
    ///
    /// assert_eq!(map.len(), 4);
    /// assert_eq!(map.len(), xs.n_unique());
    /// ```
    fn value_map(&self) -> BTreeMap<usize, Self::Item> {
        let mut map = BTreeMap::new();
        self.unique_values()
            .drain(..)
            .enumerate()
            .for_each(|(id, value)| {
                map.insert(id, value);
            });
        map
    }
}

impl<T> UniqueCollection for Vec<T>
where
    T: Hash + Eq + Clone,
{
    type Item = T;

    fn unique_values(&self) -> Vec<T> {
        let mut set = HashSet::new();
        self.into_iter().for_each(|value| {
            if !set.contains(value) {
                set.insert(value.to_owned());
            }
        });
        set.into_iter().collect()
    }

    fn n_unique_cutoff(&self, cutoff: usize) -> usize {
        let mut set = HashSet::new();
        for value in self.into_iter() {
            if set.len() == cutoff {
                return cutoff;
            }

            if !set.contains(value) {
                set.insert(value.to_owned());
            }
        }
        set.len()
    }
}
