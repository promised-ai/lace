use lace_stats::assignment::Assignment;

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

#[derive(Clone, Debug)]
pub struct RowSamsInfo {
    pub i: usize,
    pub j: usize,
    pub z_i: usize,
    pub z_j: usize,
    pub z_split: usize,
}

pub trait RowSamsConstrainer: Sync {
    /// Called first. Provides the setup info to the constrainer.
    fn initialize(&mut self, info: RowSamsInfo);

    /// The ln Sequential Importance Sampling contstraint called when
    /// proposing, or calculating the reverse transition function of, the
    /// split.
    fn ln_sis_contstraints(&mut self, ix: usize) -> (f64, f64);

    /// Assign ix to z during SIS
    fn sis_assign(&mut self, ix: usize, to_proposed_cluster: bool);

    /// Should return the log hastings ratio constraint, which is
    /// ln p(x|z_proposed) - ln p (x|z_current)
    fn ln_mh_constraint(&self, asgn_proposed: &Assignment) -> f64;
}

impl RowSamsConstrainer for () {
    fn initialize(&mut self, _info: RowSamsInfo) {}

    fn ln_sis_contstraints(&mut self, _ix: usize) -> (f64, f64) {
        (0.0, 0.0)
    }

    fn sis_assign(&mut self, _ix: usize, _to_proposed_cluster: bool) {}

    fn ln_mh_constraint(&self, _asgn_proposed: &Assignment) -> f64 {
        0.0
    }
}
