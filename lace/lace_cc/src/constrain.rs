pub trait RowConstrainer: Sync {
    fn ln_constraint(&self, row_ix: usize, k: usize) -> f64;
}

impl RowConstrainer for () {
    fn ln_constraint(&self, _row_ix: usize, _k: usize) -> f64 {
        0.0
    }
}
