use std::ops::Index;

/// A lightweight Matrix abstraction that does almost nothing.
#[derive(Clone, Debug)]
pub struct Matrix<T: Send + Sync> {
    nrows: usize,
    ncols: usize,
    values: Vec<T>,
}

impl<T: Send + Sync> Matrix<T> {
    pub fn from_raw_parts(values: Vec<T>, nrows: usize) -> Self {
        let ncols = values.len() / nrows;
        assert_eq!(values.len(), nrows * ncols);
        Matrix {
            nrows,
            ncols,
            values,
        }
    }

    /// Create a new Matrix from a vector of vectors
    pub fn from_vecs(mut vecs: Vec<Vec<T>>) -> Self {
        let nrows = vecs.len();
        let ncols = vecs[0].len();
        let mut values = Vec::with_capacity(nrows * ncols);

        vecs.drain(..).for_each(|mut row| {
            row.drain(..).for_each(|x| values.push(x));
        });

        Matrix {
            nrows,
            ncols,
            values,
        }
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

    /// Create a mutable iterator through rows
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
    /// let mut mat = Matrix::from_vecs(vecs);
    ///
    /// mat.rows_mut().for_each(|mut row| {
    ///     row.iter_mut().for_each(|mut x| *x += 1 );
    /// });
    ///
    /// assert_eq!(mat.raw_values(), &vec![1, 2, 3, 4, 5, 6])
    /// ```
    #[inline]
    pub fn rows_mut(&mut self) -> RowIterMut<T> {
        RowIterMut {
            values: &mut self.values,
            ix: 0,
            ncols: self.ncols,
            nrows: self.nrows,
        }
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
    /// let mut mat = Matrix::from_vecs(vecs);
    ///
    /// let rowsum: Vec<u8> = mat.rows().map(|row| {
    ///     row.iter().sum::<u8>()
    /// })
    /// .collect();
    ///
    /// assert_eq!(rowsum, vec![3_u8, 12_u8])
    /// ```
    #[inline]
    pub fn rows(&self) -> RowIter<T> {
        RowIter {
            values: &self.values,
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
    /// use braid_utils::Shape;
    ///
    /// let vecs: Vec<Vec<u32>> = vec![
    ///     vec![0, 1, 2],
    ///     vec![3, 4, 5],
    ///     vec![6, 7, 8],
    ///     vec![9, 10, 11],
    /// ];
    ///
    /// let mat = Matrix::from_vecs(vecs);
    /// let mat_t = mat.clone().implicit_transpose();
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
    pub fn implicit_transpose(self) -> ImplicitlyTransposedMatrix<T> {
        ImplicitlyTransposedMatrix {
            nrows: self.ncols,
            ncols: self.nrows,
            values: self.values,
        }
    }
}

impl<T: Send + Sync + Clone> Matrix<T> {
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
        col.iter().for_each(|x| {
            (0..ncols).for_each(|_| values.push(x.clone()));
        });

        Matrix {
            ncols,
            nrows,
            values,
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T>
where
    T: Send + Sync,
{
    type Output = T;

    #[inline]
    fn index(&self, ix: (usize, usize)) -> &Self::Output {
        let (i, j) = ix;
        &self.values[self.ncols * i + j]
    }
}

impl<T> Index<(usize, usize)> for &Matrix<T>
where
    T: Send + Sync,
{
    type Output = T;

    #[inline]
    fn index(&self, ix: (usize, usize)) -> &Self::Output {
        let (i, j) = ix;
        &self.values[self.ncols * i + j]
    }
}

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

/// Allows iteration through rows of a Matrix
pub struct RowIter<'a, T> {
    values: &'a Vec<T>,
    ix: usize,
    ncols: usize,
    nrows: usize,
}

impl<'a, T> Iterator for RowIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.ix == self.nrows {
            None
        } else {
            let out = unsafe {
                let ptr = self.values.as_ptr().add(self.ix * self.ncols);
                std::slice::from_raw_parts(ptr, self.ncols)
            };
            self.ix += 1;
            Some(out)
        }
    }
}

#[derive(Clone, Debug)]
pub struct ImplicitlyTransposedMatrix<T: Send + Sync> {
    nrows: usize,
    ncols: usize,
    values: Vec<T>,
}

impl<T> Index<(usize, usize)> for ImplicitlyTransposedMatrix<T>
where
    T: Send + Sync,
{
    type Output = T;

    #[inline]
    fn index(&self, ix: (usize, usize)) -> &Self::Output {
        let (i, j) = ix;
        &self.values[self.nrows * j + i]
    }
}

impl<T> Index<(usize, usize)> for &ImplicitlyTransposedMatrix<T>
where
    T: Send + Sync,
{
    type Output = T;

    #[inline]
    fn index(&self, ix: (usize, usize)) -> &Self::Output {
        let (i, j) = ix;
        &self.values[self.nrows * j + i]
    }
}

impl<T: Send + Sync> ImplicitlyTransposedMatrix<T> {
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

    pub fn transpose(self) -> Matrix<T> {
        Matrix {
            nrows: self.ncols,
            ncols: self.nrows,
            values: self.values,
        }
    }
}

pub trait Shape {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
}

impl<T: Send + Sync> Shape for Matrix<T> {
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<T: Send + Sync> Shape for &Matrix<T> {
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<T: Send + Sync> Shape for ImplicitlyTransposedMatrix<T> {
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<T: Send + Sync> Shape for &ImplicitlyTransposedMatrix<T> {
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }
}
