mod dense;
mod sparse;
mod traits;

pub use dense::DenseContainer;
pub use sparse::SparseContainer;
pub use traits::AccumScore;
pub use traits::Container;

impl<T: Clone + Default> From<DenseContainer<T>> for SparseContainer<T> {
    fn from(dense: DenseContainer<T>) -> Self {
        SparseContainer::with_missing(dense.data, &dense.present)
    }
}

impl<T: Clone + Default> From<SparseContainer<T>> for DenseContainer<T> {
    fn from(mut sparse: SparseContainer<T>) -> Self {
        let mut data: Vec<T> = Vec::new();
        let mut present: Vec<bool> = Vec::new();

        let mut last_ix = 0;
        let n = sparse.len();

        if n == 0 {
            return DenseContainer { data, present };
        }

        sparse.data.drain(..).for_each(|(ix, xs)| {
            if ix > last_ix {
                let n_interval = ix - last_ix;
                data.extend(vec![T::default(); n_interval]);
                present.extend(vec![false; n_interval]);
            }
            last_ix = ix + xs.len();
            present.extend(vec![true; xs.len()]);
            data.extend(xs);
        });

        if last_ix < n {
            let n_interval = n - last_ix;
            data.extend(vec![T::default(); n_interval]);
            present.extend(vec![false; n_interval]);
        }

        assert_eq!(data.len(), n);
        assert_eq!(present.len(), n);

        DenseContainer { data, present }
    }
}
