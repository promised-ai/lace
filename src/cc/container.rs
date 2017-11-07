use std::ops::{Index, IndexMut};
use cc::assignment::Assignment;


#[allow(dead_code)]
pub struct DataContainer<T> where T: Clone {
    pub data: Vec<T>,
    pub present: Vec<bool>,
}

impl<T> DataContainer<T> where T: Clone {
    pub fn new(data: Vec<T>) -> DataContainer<T> {
        let n = data.len();
        DataContainer{data: data, present: vec![true; n]}
    }

    pub fn with_filter<F>(mut data: Vec<T>, defval: T, pred: F) -> DataContainer<T>
        where F: Fn(&T) -> bool
    {
        let n = data.len();
        let mut present: Vec<bool> = vec![true; n];
        for i in 0..n {
            if !pred(&data[i]) {
                present[i] = false;
                data[i] = defval.clone();
            }
        }
        DataContainer{data: data, present: present}
    }

    // TODO: Add method to construct sufficient statistics
    // XXX: might be faster to use nested for loop?
    pub fn group_by<'a>(&self, asgn: &'a Assignment) -> Vec<Vec<&T>> {
        // FIXME: Filter on `present` using better zip library
        (0..asgn.ncats).map(|k| {
            self.data.iter()
                     .zip(asgn.asgn.iter())
                     .filter(|&(_, z)| *z == k)
                     .map(|(x, _)| x)
                     .collect()
        }).collect()
    }
}


impl<T> Index<usize> for DataContainer<T> where T: Clone {
    type Output = T;
    fn index(&self, ix: usize) -> &T {
        & self.data[ix]
    }
}


impl<T> IndexMut<usize> for DataContainer<T> where T: Clone {
    fn index_mut<'a>(&'a mut self, ix: usize) -> &'a mut T {
        &mut self.data[ix]
    }
}
