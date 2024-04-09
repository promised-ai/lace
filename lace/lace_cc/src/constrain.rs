pub trait RowConstrainer: Sync {
    fn ln_constraint(&self, row_ix: usize, k: usize) -> f64;
}

impl RowConstrainer for () {
    fn ln_constraint(&self, _row_ix: usize, _k: usize) -> f64 {
        0.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RowGibbsInfo {
    pub is_singleton: bool,
    pub k_original: usize,
}

impl Default for RowGibbsInfo {
    fn default() -> Self {
        Self {
            is_singleton: false,
            k_original: 0,
        }
    }
}

pub trait RowGibbsConstrainer: Sync {
    fn ln_constraint(&self, info: RowGibbsInfo, row_ix: usize, k: usize)
        -> f64;
}

impl RowGibbsConstrainer for () {
    fn ln_constraint(
        &self,
        _info: RowGibbsInfo,
        _row_ix: usize,
        _k: usize,
    ) -> f64 {
        0.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RowSamsInfo {
    pub propose_merge: bool,
    pub z_i: usize,
    pub z_j: usize,
}

impl Default for RowSamsInfo {
    fn default() -> Self {
        Self {
            propose_merge: false,
            z_i: 0,
            z_j: 0,
        }
    }
}

pub trait RowSamsConstrainer: Sync {
    fn ln_constraint(&self, info: RowSamsInfo, row_ix: usize, k: usize) -> f64;
}

impl RowSamsConstrainer for () {
    fn ln_constraint(
        &self,
        _info: RowSamsInfo,
        _row_ix: usize,
        _k: usize,
    ) -> f64 {
        0.0
    }
}
