use std::ops::Index;

/// A lightweight Matrix abstraction that does almost nothing.
#[derive(Clone, Debug)]
pub struct Matrix<T: Send + Sync> {
    n_rows: usize,
    n_cols: usize,
    values: Vec<T>,
}

impl<T: Send + Sync> Matrix<T> {
    pub fn from_raw_parts(values: Vec<T>, n_rows: usize) -> Self {
        let n_cols = values.len() / n_rows;
        assert_eq!(values.len(), n_rows * n_cols);
        Matrix {
            n_rows,
            n_cols,
            values,
        }
    }

    /// Create a new Matrix from a vector of vectors
    pub fn from_vecs(mut vecs: Vec<Vec<T>>) -> Self {
        let n_rows = vecs.len();
        let n_cols = vecs[0].len();
        let mut values = Vec::with_capacity(n_rows * n_cols);

        vecs.drain(..).for_each(|mut row| {
            row.drain(..).for_each(|x| values.push(x));
        });

        Matrix {
            n_rows,
            n_cols,
            values,
        }
    }

    #[inline]
    pub fn nelem(&self) -> usize {
        self.n_cols * self.n_rows
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
    pub fn rows_mut(&mut self) -> std::slice::ChunksMut<T> {
        self.values.chunks_mut(self.n_cols)
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
    pub fn rows(&self) -> std::slice::Chunks<T> {
        self.values.chunks(self.n_cols)
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
    /// assert_eq!(mat.n_rows(), mat_t.n_cols());
    /// assert_eq!(mat.n_cols(), mat_t.n_rows());
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
            n_rows: self.n_cols,
            n_cols: self.n_rows,
            values: self.values,
        }
    }
}

impl<T: Send + Sync + Clone> Matrix<T> {
    /// Treat the input vector, `col` like a column vector and replicate it
    /// `n_cols` times.
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
    pub fn vtile(col: Vec<T>, n_cols: usize) -> Self {
        let n_rows = col.len();
        let mut values: Vec<T> = Vec::with_capacity(n_rows * n_cols);
        col.iter().for_each(|x| {
            (0..n_cols).for_each(|_| values.push(x.clone()));
        });

        Matrix {
            n_rows,
            n_cols,
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
        &self.values[self.n_cols * i + j]
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
        &self.values[self.n_cols * i + j]
    }
}

#[derive(Clone, Debug)]
pub struct ImplicitlyTransposedMatrix<T: Send + Sync> {
    n_rows: usize,
    n_cols: usize,
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
        &self.values[self.n_rows * j + i]
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
        &self.values[self.n_rows * j + i]
    }
}

impl<T: Send + Sync> ImplicitlyTransposedMatrix<T> {
    #[inline]
    pub fn nelem(&self) -> usize {
        self.n_cols * self.n_rows
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
            n_rows: self.n_cols,
            n_cols: self.n_rows,
            values: self.values,
        }
    }
}

pub trait Shape {
    fn n_rows(&self) -> usize;
    fn n_cols(&self) -> usize;
    fn shape(&self) -> (usize, usize) {
        (self.n_rows(), self.n_cols())
    }
}

impl<T: Send + Sync> Shape for Matrix<T> {
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn n_cols(&self) -> usize {
        self.n_cols
    }
}

impl<T: Send + Sync> Shape for &Matrix<T> {
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn n_cols(&self) -> usize {
        self.n_cols
    }
}

impl<T: Send + Sync> Shape for ImplicitlyTransposedMatrix<T> {
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn n_cols(&self) -> usize {
        self.n_cols
    }
}

impl<T: Send + Sync> Shape for &ImplicitlyTransposedMatrix<T> {
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn n_cols(&self) -> usize {
        self.n_cols
    }
}
