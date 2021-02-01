use std::ops::Mul;

#[derive(Clone, Debug)]
pub struct Vector2(pub [f64; 2]);

#[derive(Clone, Debug)]
pub struct Vector4(pub [f64; 4]);

#[derive(Clone, Debug)]
pub struct Matrix1x1(pub [f64; 1]);

#[derive(Clone, Debug)]
pub struct Matrix2x2(pub [f64; 4]);

#[derive(Clone, Debug)]
pub struct Matrix4x4(pub [f64; 16]);

impl Vector2 {
    pub fn new() -> Self {
        Vector2([0.0; 2])
    }
}

impl Vector4 {
    pub fn new() -> Self {
        Vector4([0.0; 4])
    }
}

impl Matrix1x1 {
    pub fn new() -> Self {
        Matrix1x1([0.0])
    }
}

impl Matrix2x2 {
    pub fn new() -> Self {
        Matrix2x2([0.0; 4])
    }

    pub fn from_diag(diag: [f64; 2]) -> Self {
        let mut mat = Self::new();
        mat.0[0] = diag[0];
        mat.0[3] = diag[1];
        mat
    }
}

impl Matrix4x4 {
    pub fn new() -> Self {
        Matrix4x4([0.0; 16])
    }

    pub fn from_diag(diag: [f64; 4]) -> Self {
        let mut mat = Self::new();
        mat.0[0] = diag[0];
        mat.0[5] = diag[1];
        mat.0[11] = diag[2];
        mat.0[15] = diag[3];
        mat
    }
}

pub trait MvSub {
    fn mv_sub(self, other: &Self) -> Self;
}

impl MvSub for Matrix1x1 {
    fn mv_sub(mut self, other: &Self) -> Self {
        self.0[0] -= other.0[0];
        self
    }
}

impl MvSub for Vector2 {
    fn mv_sub(mut self, other: &Self) -> Self {
        self.0[0] -= other.0[0];
        self.0[1] -= other.0[1];
        self
    }
}

impl MvSub for Matrix2x2 {
    fn mv_sub(mut self, other: &Self) -> Self {
        self.0[0] -= other.0[0];
        self.0[1] -= other.0[1];
        self.0[2] -= other.0[2];
        self.0[3] -= other.0[3];
        self
    }
}

impl MvSub for Vector4 {
    fn mv_sub(mut self, other: &Self) -> Self {
        self.0[0] -= other.0[0];
        self.0[1] -= other.0[1];
        self.0[2] -= other.0[2];
        self.0[3] -= other.0[3];
        self
    }
}

impl MvSub for Matrix4x4 {
    fn mv_sub(mut self, other: &Self) -> Self {
        self.0[0] -= other.0[0];
        self.0[1] -= other.0[1];
        self.0[2] -= other.0[2];
        self.0[3] -= other.0[3];
        self.0[4] -= other.0[4];
        self.0[5] -= other.0[5];
        self.0[6] -= other.0[6];
        self.0[7] -= other.0[7];
        self.0[8] -= other.0[8];
        self.0[9] -= other.0[9];
        self.0[10] -= other.0[10];
        self.0[11] -= other.0[11];
        self.0[12] -= other.0[12];
        self.0[13] -= other.0[13];
        self.0[14] -= other.0[14];
        self.0[15] -= other.0[15];
        self
    }
}

pub trait MvAdd {
    fn mv_add(self, other: &Self) -> Self;
}

impl MvAdd for Matrix1x1 {
    fn mv_add(mut self, other: &Self) -> Self {
        self.0[0] += other.0[0];
        self
    }
}

impl MvAdd for Vector2 {
    fn mv_add(mut self, other: &Self) -> Self {
        self.0[0] += other.0[0];
        self.0[1] += other.0[1];
        self
    }
}

impl MvAdd for Matrix2x2 {
    fn mv_add(mut self, other: &Self) -> Self {
        self.0[0] += other.0[0];
        self.0[1] += other.0[1];
        self.0[2] += other.0[2];
        self.0[3] += other.0[3];
        self
    }
}

impl MvAdd for Vector4 {
    fn mv_add(mut self, other: &Self) -> Self {
        self.0[0] += other.0[0];
        self.0[1] += other.0[1];
        self.0[2] += other.0[2];
        self.0[3] += other.0[3];
        self
    }
}

impl MvAdd for Matrix4x4 {
    fn mv_add(mut self, other: &Self) -> Self {
        self.0[0] += other.0[0];
        self.0[1] += other.0[1];
        self.0[2] += other.0[2];
        self.0[3] += other.0[3];
        self.0[4] += other.0[4];
        self.0[5] += other.0[5];
        self.0[6] += other.0[6];
        self.0[7] += other.0[7];
        self.0[8] += other.0[8];
        self.0[9] += other.0[9];
        self.0[10] += other.0[10];
        self.0[11] += other.0[11];
        self.0[12] += other.0[12];
        self.0[13] += other.0[13];
        self.0[14] += other.0[14];
        self.0[15] += other.0[15];
        self
    }
}

impl Mul<f64> for Matrix1x1 {
    type Output = Matrix1x1;

    fn mul(mut self, rhs: f64) -> Self::Output {
        self.0[0] *= rhs;
        self
    }
}

impl Mul<f64> for Vector2 {
    type Output = Vector2;

    fn mul(mut self, rhs: f64) -> Self::Output {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
        self
    }
}

impl Mul<f64> for Vector4 {
    type Output = Vector4;

    fn mul(mut self, rhs: f64) -> Self::Output {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
        self.0[2] *= rhs;
        self.0[3] *= rhs;
        self
    }
}

impl Mul<f64> for Matrix2x2 {
    type Output = Matrix2x2;

    fn mul(mut self, rhs: f64) -> Self::Output {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
        self.0[2] *= rhs;
        self.0[3] *= rhs;
        self
    }
}

impl Mul<f64> for Matrix4x4 {
    type Output = Matrix4x4;

    fn mul(mut self, rhs: f64) -> Self::Output {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
        self.0[2] *= rhs;
        self.0[3] *= rhs;
        self.0[4] *= rhs;
        self.0[5] *= rhs;
        self.0[6] *= rhs;
        self.0[7] *= rhs;
        self.0[8] *= rhs;
        self.0[9] *= rhs;
        self.0[10] *= rhs;
        self.0[11] *= rhs;
        self.0[12] *= rhs;
        self.0[13] *= rhs;
        self.0[14] *= rhs;
        self.0[15] *= rhs;
        self
    }
}

/// Squares a vector x with x * x^T
pub trait SquareT {
    type Output;
    fn square_t(&self) -> Self::Output;
}

#[rustfmt::skip]
impl SquareT for Matrix1x1 {
    type Output = Matrix1x1;

    fn square_t(&self) -> Self::Output {
        Matrix1x1([self.0[0].powi(2)])
    }
}

#[rustfmt::skip]
impl SquareT for Vector2 {
    type Output = Matrix2x2;

    fn square_t(&self) -> Matrix2x2 {
        let [x1, x2] = self.0;

        Matrix2x2 ([
            x1*x1, x1*x2,
            x2*x1, x2*x2,
        ])
    }
}

#[rustfmt::skip]
impl SquareT for Vector4 {
    type Output = Matrix4x4;

    fn square_t(&self) -> Matrix4x4 {
        let [x1, x2, x3, x4] = self.0;

        Matrix4x4 ([
            x1*x1, x1*x2, x1*x3, x1*x4,
            x2*x1, x2*x2, x2*x3, x2*x4,
            x3*x1, x3*x2, x3*x3, x3*x4,
            x4*x1, x4*x2, x4*x3, x4*x4,
        ])
    }
}

pub trait MeanVector: Mul<f64> + MvAdd + MvSub + SquareT + Clone {
    fn values(&self) -> &[f64];

    fn len(&self) -> usize;

    fn from_dvector(vec: nalgebra::DVector<f64>) -> Self;

    fn zeros() -> Self;
}

pub trait ScaleMatrix: Mul<f64> + MvAdd + MvSub + Clone {
    fn values(&self) -> &[f64];

    /// Set all off-diagonal elements to zero
    fn diagonalize(&mut self);
}

impl MeanVector for Matrix1x1 {
    fn values(&self) -> &[f64] {
        &self.0
    }

    fn len(&self) -> usize {
        1
    }

    fn from_dvector(vec: nalgebra::DVector<f64>) -> Self {
        Matrix1x1([vec[0]])
    }

    fn zeros() -> Self {
        Matrix1x1([0.0])
    }
}

impl MeanVector for Vector2 {
    fn values(&self) -> &[f64] {
        &self.0
    }

    fn len(&self) -> usize {
        2
    }

    fn from_dvector(vec: nalgebra::DVector<f64>) -> Self {
        Vector2([vec[0], vec[1]])
    }

    fn zeros() -> Self {
        Self::new()
    }
}

impl MeanVector for Vector4 {
    fn values(&self) -> &[f64] {
        &self.0
    }

    fn len(&self) -> usize {
        4
    }

    fn from_dvector(vec: nalgebra::DVector<f64>) -> Self {
        Vector4([vec[0], vec[1], vec[2], vec[3]])
    }

    fn zeros() -> Self {
        Self::new()
    }
}

impl ScaleMatrix for Matrix2x2 {
    fn values(&self) -> &[f64] {
        &self.0
    }

    fn diagonalize(&mut self) {
        self.0[1] = 0.0;
        self.0[2] = 0.0;
    }
}

impl ScaleMatrix for Matrix4x4 {
    fn values(&self) -> &[f64] {
        &self.0
    }

    fn diagonalize(&mut self) {
        self.0[1] = 0.0;
        self.0[2] = 0.0;
        self.0[3] = 0.0;

        self.0[4] = 0.0;
        self.0[6] = 0.0;
        self.0[7] = 0.0;

        self.0[8] = 0.0;
        self.0[9] = 0.0;
        self.0[11] = 0.0;

        self.0[12] = 0.0;
        self.0[13] = 0.0;
        self.0[14] = 0.0;
    }
}

impl ScaleMatrix for Matrix1x1 {
    fn values(&self) -> &[f64] {
        &self.0
    }

    fn diagonalize(&mut self) {}
}
