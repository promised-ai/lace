use std::ops::Index;

/// Allows mutable iteration through rows of a Matrix
pub struct RowIterMut<'a, T> {
    values: &'a mut Vec<T>,
    ix: usize,
    ncols: usize,
    nrows: usize,
}

impl<'a, T> Iterator for RowIterMut<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.ix == self.nrows {
            None
        } else {
            let out = unsafe {
                let ptr = self.values.as_mut_ptr().add(self.ix * self.ncols);
                std::slice::from_raw_parts_mut(ptr, self.ncols)
            };
            self.ix += 1;
            Some(out)
        }
    }
}

/// A lightweight Matrix abstraction that does almost nothing.
#[derive(Clone, Debug)]
pub struct Matrix<T: Copy + Send + Sync> {
    nrows: usize,
    ncols: usize,
    values: Vec<T>,
    transpose: bool,
}

impl<T: Copy + Send + Sync> Matrix<T> {
    pub fn from_raw_parts(values: Vec<T>, nrows: usize) -> Self {
        let ncols = values.len() / nrows;
        assert_eq!(values.len(), nrows * ncols);
        Matrix {
            nrows,
            ncols,
            values,
            transpose: false,
        }
    }

    /// Treat the input vector, `col` like a column vector and replicate it
    /// `ncols` times.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use braid_utils::Matrix;
    /// let col: Vec<u32> = vec![0, 1, 2];
    ///
    /// let mat = Matrix::vtile(col, 12);
    ///
    /// assert_eq!(mat[(0, 0)], 0);
    /// assert_eq!(mat[(0, 11)], 0);
    ///
    /// assert_eq!(mat[(1, 0)], 1);
    /// assert_eq!(mat[(1, 11)], 1);
    ///
    /// assert_eq!(mat[(2, 0)], 2);
    /// assert_eq!(mat[(2, 11)], 2);
    /// ```
    pub fn vtile(col: Vec<T>, ncols: usize) -> Self {
        let nrows = col.len();
        let mut values: Vec<T> = Vec::with_capacity(nrows * ncols);
        col.iter().for_each(|&x| {
            (0..ncols).for_each(|_| values.push(x));
        });

        Matrix {
            ncols,
            nrows,
            values,
            transpose: false,
        }
    }

    /// Create a new Matrix from a vector of vectors
    pub fn from_vecs(vecs: &Vec<Vec<T>>) -> Self {
        let nrows = vecs.len();
        let ncols = vecs[0].len();
        let mut values = Vec::with_capacity(nrows * ncols);
        vecs.iter().for_each(|row| {
            row.iter().for_each(|&x| values.push(x));
        });
        Matrix {
            nrows,
            ncols,
            values,
            transpose: false,
        }
    }

    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline]
    pub fn nelem(&self) -> usize {
        self.ncols * self.nrows
    }

    #[inline]
    pub fn raw_values(&self) -> &Vec<T> {
        &self.values
    }

    #[inline]
    pub fn raw_values_mut(&mut self) -> &mut Vec<T> {
        &mut self.values
    }

    /// Create an iterator through rows
    ///
    /// # Example
    ///
    /// ```rust
    /// # use braid_utils::Matrix;
    /// let vecs: Vec<Vec<u8>> = vec![
    ///     vec![0, 1, 2],
    ///     vec![3, 4, 5],
    /// ];
    ///
    /// let mut mat = Matrix::from_vecs(&vecs);
    ///
    /// mat.rows_mut().for_each(|mut row| {
    ///     row.iter_mut().for_each(|mut x| *x += 1 );
    /// });
    ///
    /// assert_eq!(mat.raw_values(), &vec![1, 2, 3, 4, 5, 6])
    ///
    /// ```
    #[inline]
    pub fn rows_mut(&mut self) -> RowIterMut<T> {
        if self.transpose {
            panic!("cannot call rows on transposed Matrix");
        }

        RowIterMut {
            values: &mut self.values,
            ix: 0,
            ncols: self.ncols,
            nrows: self.nrows,
        }
    }

    /// Does an implicit transpose by inverting coordinates.
    ///
    /// # Notes
    /// The matrix is not rebuild, so if you are attempting to access the
    /// transposed matrix row-wise many times, there will be a lot of cache
    /// misses. This method is best used if you plan to traverse the transposed
    /// matrix a small number of times.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use braid_utils::Matrix;
    /// let vecs: Vec<Vec<u32>> = vec![
    ///     vec![0, 1, 2],
    ///     vec![3, 4, 5],
    ///     vec![6, 7, 8],
    ///     vec![9, 10, 11],
    /// ];
    ///
    /// let mat = Matrix::from_vecs(&vecs);
    /// let mat_t = {
    ///     let mut mat_t = mat.clone();
    ///     mat_t.implicit_transpose();
    ///     mat_t
    /// };
    ///
    /// assert_eq!(mat.nrows(), mat_t.ncols());
    /// assert_eq!(mat.ncols(), mat_t.nrows());
    ///
    /// for i in 0..3 {
    ///     for j in 0..3 {
    ///         assert_eq!(mat[(i, j)], mat_t[(j, i)]);
    ///     }
    /// }
    ///
    /// ```
    #[inline]
    pub fn implicit_transpose(&mut self) {
        std::mem::swap(&mut self.nrows, &mut self.ncols);
        self.transpose = !self.transpose;
    }
}

impl<T> Index<(usize, usize)> for Matrix<T>
where
    T: Copy + Send + Sync,
{
    type Output = T;

    #[inline]
    fn index(&self, ix: (usize, usize)) -> &Self::Output {
        let (i, j) = ix;
        if self.transpose {
            &self.values[self.nrows * j + i]
        } else {
            &self.values[self.ncols * i + j]
        }
    }
}
