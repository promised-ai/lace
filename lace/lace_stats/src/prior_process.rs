use lace_consts::rv::{misc::pflip, traits::Rv};
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::assignment::{Assignment, AssignmentError};
use crate::rv::dist::{Beta, Gamma};

const MAX_STICK_BREAKING_ITERS: u16 = 10_000;

pub trait PriorProcessT {
    fn ln_gibbs_weight(&self, n_k: usize) -> f64;

    fn ln_singleton_weight(&self, n_cats: usize) -> f64;

    fn weight_vec(
        &self,
        asgn: &Assignment,
        normed: bool,
        append_new: bool,
    ) -> Vec<f64>;

    fn slice_sb_extend<R: Rng>(
        &self,
        weights: Vec<f64>,
        u_star: f64,
        rng: &mut R,
    ) -> Vec<f64>;

    fn draw_assignment<R: Rng>(&self, n: usize, rng: &mut R) -> Assignment;

    fn update_params<R: Rng>(&mut self, asgn: &Assignment, rng: &mut R) -> f64;

    fn reset_params<R: Rng>(&mut self, rng: &mut R);
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Dirichlet {
    pub alpha: f64,
    pub prior: Gamma,
}

impl Dirichlet {
    pub fn from_prior<R: Rng>(prior: Gamma, rng: &mut R) -> Self {
        Self {
            alpha: prior.draw(rng),
            prior,
        }
    }
}

impl PriorProcessT for Dirichlet {
    fn ln_gibbs_weight(&self, n_k: usize) -> f64 {
        (n_k as f64).ln()
    }

    fn ln_singleton_weight(&self, _n_cats: usize) -> f64 {
        self.alpha.ln()
    }

    fn weight_vec(
        &self,
        asgn: &Assignment,
        normed: bool,
        append_new: bool,
    ) -> Vec<f64> {
        let mut weights: Vec<f64> =
            asgn.counts.iter().map(|&ct| ct as f64).collect();

        let z = if append_new {
            weights.push(self.alpha);
            asgn.len() as f64 + self.alpha
        } else {
            asgn.len() as f64
        };

        if normed {
            weights.iter_mut().for_each(|ct| *ct /= z);
        }

        weights
    }

    fn slice_sb_extend<R: Rng>(
        &self,
        weights: Vec<f64>,
        u_star: f64,
        rng: &mut R,
    ) -> Vec<f64> {
        sb_slice_extend(weights, self.alpha, 0.0, u_star, rng).unwrap()
    }

    fn draw_assignment<R: Rng>(&self, n: usize, rng: &mut R) -> Assignment {
        if n == 0 {
            return Assignment::empty();
        }
        let mut counts = vec![1];
        let mut ps = vec![1.0, self.alpha];
        let mut zs = vec![0; n];

        for i in 1..n {
            let zi = pflip(&ps, 1, rng)[0];
            zs[i] = zi;
            if zi < counts.len() {
                ps[zi] += 1.0;
                counts[zi] += 1;
            } else {
                ps[zi] = 1.0;
                ps.push(self.alpha);
                counts.push(1);
            };
        }

        Assignment {
            asgn: zs,
            n_cats: counts.len(),
            counts,
        }
    }

    fn update_params<R: Rng>(&mut self, asgn: &Assignment, rng: &mut R) -> f64 {
        // TODO: Should we use a different method to draw CRP alpha that can
        // extend outside of the bulk of the prior's mass?
        let cts = &asgn.counts;
        let n: usize = asgn.len();
        let loglike = |alpha: &f64| crate::assignment::lcrp(n, cts, *alpha);
        let prior_ref = &self.prior;
        let prior_draw = |rng: &mut R| prior_ref.draw(rng);
        let mh_result =
            crate::mh::mh_prior(self.alpha, loglike, prior_draw, 100, rng);
        self.alpha = mh_result.x;
        mh_result.score_x
    }

    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        self.alpha = self.prior.draw(rng);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PitmanYor {
    pub alpha: f64,
    pub d: f64,
    pub prior_alpha: Gamma,
    pub prior_d: Beta,
}

impl PitmanYor {
    pub fn from_prior<R: Rng>(
        prior_alpha: Gamma,
        prior_d: Beta,
        rng: &mut R,
    ) -> Self {
        Self {
            alpha: prior_alpha.draw(rng),
            d: prior_d.draw(rng),
            prior_alpha,
            prior_d,
        }
    }
}

impl PriorProcessT for PitmanYor {
    fn ln_gibbs_weight(&self, n_k: usize) -> f64 {
        (n_k as f64 - self.d).ln()
    }

    fn ln_singleton_weight(&self, n_cats: usize) -> f64 {
        (self.alpha + self.d * (n_cats as f64)).ln()
    }

    fn weight_vec(
        &self,
        asgn: &Assignment,
        normed: bool,
        append_new: bool,
    ) -> Vec<f64> {
        let mut weights: Vec<f64> =
            asgn.counts.iter().map(|&ct| ct as f64 - self.d).collect();

        let z = if append_new {
            weights.push(self.alpha + self.d * asgn.n_cats as f64);
            asgn.len() as f64 + self.alpha
        } else {
            asgn.len() as f64
        };

        if normed {
            weights.iter_mut().for_each(|ct| *ct /= z);
        }

        weights
    }

    fn slice_sb_extend<R: Rng>(
        &self,
        weights: Vec<f64>,
        u_star: f64,
        rng: &mut R,
    ) -> Vec<f64> {
        sb_slice_extend(weights, self.alpha, self.d, u_star, rng).unwrap()
    }

    fn draw_assignment<R: Rng>(&self, n: usize, rng: &mut R) -> Assignment {
        if n == 0 {
            return Assignment::empty();
        }

        let mut counts = vec![1];
        let mut ps = vec![1.0 - self.d, self.alpha + self.d];
        let mut zs = vec![0; n];

        for i in 1..n {
            let zi = pflip(&ps, 1, rng)[0];
            zs[i] = zi;
            if zi < counts.len() {
                ps[zi] += 1.0;
                counts[zi] += 1;
            } else {
                ps[zi] = 1.0;
                counts.push(1);
                ps.push(self.alpha + self.d * counts.len() as f64);
            };
        }

        Assignment {
            asgn: zs,
            n_cats: counts.len(),
            counts,
        }
    }

    fn update_params<R: Rng>(&mut self, asgn: &Assignment, rng: &mut R) -> f64 {
        let cts = &asgn.counts;
        let n: usize = asgn.len();
        // TODO: this is not the best way to do this.
        let ln_f_alpha = {
            let loglike = |alpha: &f64| crate::assignment::lcrp(n, cts, *alpha);
            let prior_ref = &self.prior_alpha;
            let prior_draw = |rng: &mut R| prior_ref.draw(rng);
            let mh_result =
                crate::mh::mh_prior(self.alpha, loglike, prior_draw, 100, rng);
            self.alpha = mh_result.x;
            mh_result.score_x
        };

        let ln_f_d = {
            let loglike = |alpha: &f64| crate::assignment::lcrp(n, cts, *alpha);
            let prior_ref = &self.prior_alpha;
            let prior_draw = |rng: &mut R| prior_ref.draw(rng);
            let mh_result =
                crate::mh::mh_prior(self.alpha, loglike, prior_draw, 100, rng);
            self.d = mh_result.x;
            mh_result.score_x
        };

        ln_f_alpha + ln_f_d
    }

    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        self.alpha = self.prior_alpha.draw(rng);
        self.d = self.prior_d.draw(rng);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Process {
    Dirichlet(Dirichlet),
    PitmanYor(PitmanYor),
}

impl PriorProcessT for Process {
    fn ln_gibbs_weight(&self, n_k: usize) -> f64 {
        match self {
            Self::Dirichlet(proc) => proc.ln_gibbs_weight(n_k),
            Self::PitmanYor(proc) => proc.ln_gibbs_weight(n_k),
        }
    }

    fn ln_singleton_weight(&self, n_cats: usize) -> f64 {
        match self {
            Self::Dirichlet(proc) => proc.ln_singleton_weight(n_cats),
            Self::PitmanYor(proc) => proc.ln_singleton_weight(n_cats),
        }
    }

    fn weight_vec(
        &self,
        asgn: &Assignment,
        normed: bool,
        append_new: bool,
    ) -> Vec<f64> {
        match self {
            Self::Dirichlet(proc) => proc.weight_vec(asgn, normed, append_new),
            Self::PitmanYor(proc) => proc.weight_vec(asgn, normed, append_new),
        }
    }

    fn slice_sb_extend<R: Rng>(
        &self,
        weights: Vec<f64>,
        u_star: f64,
        rng: &mut R,
    ) -> Vec<f64> {
        match self {
            Self::Dirichlet(proc) => proc.slice_sb_extend(weights, u_star, rng),
            Self::PitmanYor(proc) => proc.slice_sb_extend(weights, u_star, rng),
        }
    }

    fn draw_assignment<R: Rng>(&self, n: usize, rng: &mut R) -> Assignment {
        match self {
            Self::Dirichlet(proc) => proc.draw_assignment(n, rng),
            Self::PitmanYor(proc) => proc.draw_assignment(n, rng),
        }
    }

    fn update_params<R: Rng>(&mut self, asgn: &Assignment, rng: &mut R) -> f64 {
        match self {
            Self::Dirichlet(proc) => proc.update_params(asgn, rng),
            Self::PitmanYor(proc) => proc.update_params(asgn, rng),
        }
    }

    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        match self {
            Self::Dirichlet(proc) => proc.reset_params(rng),
            Self::PitmanYor(proc) => proc.reset_params(rng),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PriorProcess {
    pub process: Process,
    pub asgn: Assignment,
}

impl PriorProcess {
    pub fn from_process<R: Rng>(
        process: Process,
        n: usize,
        rng: &mut R,
    ) -> Self {
        let asgn = process.draw_assignment(n, rng);
        Self { process, asgn }
    }

    pub fn weight_vec(&self, append_new: bool) -> Vec<f64> {
        self.process.weight_vec(&self.asgn, true, append_new)
    }

    pub fn weight_vec_unnormed(&self, append_new: bool) -> Vec<f64> {
        self.process.weight_vec(&self.asgn, false, append_new)
    }

    pub fn update_params<R: Rng>(&mut self, rng: &mut R) -> f64 {
        self.process.update_params(&self.asgn, rng)
    }

    pub fn alpha(&self) -> Option<f64> {
        match &self.process {
            Process::Dirichlet(inner) => Some(inner.alpha),
            Process::PitmanYor(inner) => Some(inner.alpha),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum PriorProcessType {
    Dirichlet,
    PitmanYor,
}

/// The stick breaking algorithm has exceeded the max number of iterations.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TheStickIsDust(u16);

// Append new dirchlet weights by stick breaking until the new weight is less
// than u*
//
// **NOTE** This function is only for the slice reassignment kernel. It cuts out
// all weights that are less that u*, so the sum of the weights will not be 1.
fn sb_slice_extend<R: Rng>(
    mut weights: Vec<f64>,
    alpha: f64,
    d: f64,
    u_star: f64,
    mut rng: &mut R,
) -> Result<Vec<f64>, TheStickIsDust> {
    let mut b_star = weights.pop().unwrap();

    // If α is low and we do the dirichlet update w ~ Dir(n_1, ..., n_k, α),
    // the final weight will often be zero. In that case, we're done.
    if b_star <= 1E-16 {
        weights.push(b_star);
        return Ok(weights);
    }

    let mut beta = Beta::new(1.0 + d, alpha).unwrap();

    let mut iters: u16 = 0;
    loop {
        if d > 0.0 {
            let n_cats = weights.len() as f64;
            beta.set_beta(alpha + d * n_cats).unwrap();
        }

        let vk: f64 = beta.draw(&mut rng);
        let bk = vk * b_star;
        b_star *= 1.0 - vk;

        if bk >= u_star {
            weights.push(bk);
        }

        if b_star < u_star {
            return Ok(weights);
        }

        iters += 1;
        if iters > MAX_STICK_BREAKING_ITERS {
            // return Err(TheStickIsDust(MAX_STICK_BREAKING_ITERS));
            eprintln!(
                "The stick is dust, n_cats: {}, u*: {}",
                weights.len(),
                u_star
            );
            return Ok(weights);
        }
    }
}

/// Constructs a PriorProcess
#[derive(Clone, Debug)]
pub struct Builder {
    n: usize,
    asgn: Option<Vec<usize>>,
    process: Option<Process>,
    seed: Option<u64>,
}

#[derive(Debug, Error, PartialEq)]
pub enum BuildPriorProcessError {
    #[error("assignment vector is empty")]
    EmptyAssignmentVec,
    #[error("there are {n_cats} categories but {n} data")]
    NLessThanNCats { n: usize, n_cats: usize },
    #[error("invalid assignment: {0}")]
    AssignmentError(#[from] AssignmentError),
}

impl Builder {
    /// Create a builder for `n`-length assignments
    ///
    /// # Arguments
    /// - n: the number of data/entries in the assignment
    pub fn new(n: usize) -> Self {
        Self {
            n,
            asgn: None,
            process: None,
            seed: None,
        }
    }

    /// Initialize the builder from an assignment vector
    ///
    /// # Note:
    /// The validity of `asgn` will not be verified until `build` is called.
    pub fn from_vec(asgn: Vec<usize>) -> Self {
        Self {
            n: asgn.len(),
            asgn: Some(asgn),
            process: None,
            seed: None,
        }
    }

    /// Select the process type
    #[must_use]
    pub fn with_process(mut self, process: Process) -> Self {
        self.process = Some(process);
        self
    }

    /// Set the RNG seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the RNG seed from another RNG
    #[must_use]
    pub fn seed_from_rng<R: rand::Rng>(mut self, rng: &mut R) -> Self {
        self.seed = Some(rng.next_u64());
        self
    }

    /// Use a *flat* assignment with one partition
    #[must_use]
    pub fn flat(mut self) -> Self {
        self.asgn = Some(vec![0; self.n]);
        self
    }

    /// Use an assignment with `n_cats`, evenly populated partitions/categories
    pub fn with_n_cats(
        mut self,
        n_cats: usize,
    ) -> Result<Self, BuildPriorProcessError> {
        if n_cats > self.n {
            Err(BuildPriorProcessError::NLessThanNCats { n: self.n, n_cats })
        } else {
            let asgn: Vec<usize> = (0..self.n).map(|i| i % n_cats).collect();
            self.asgn = Some(asgn);
            Ok(self)
        }
    }

    /// Build the assignment and consume the builder
    pub fn build(self) -> Result<PriorProcess, BuildPriorProcessError> {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = self
            .seed
            .map_or_else(|| StdRng::from_entropy(), StdRng::seed_from_u64);

        let process = self.process.unwrap_or_else(|| {
            Process::Dirichlet(Dirichlet::from_prior(
                lace_consts::general_alpha_prior(),
                &mut rng,
            ))
        });

        let n = self.n;
        let asgn = self
            .asgn
            .unwrap_or_else(|| process.draw_assignment(n, &mut rng).asgn);

        let n_cats: usize = asgn.iter().max().map(|&m| m + 1).unwrap_or(0);
        let mut counts: Vec<usize> = vec![0; n_cats];
        for z in &asgn {
            counts[*z] += 1;
        }

        let asgn = Assignment {
            asgn,
            counts,
            n_cats,
        };

        if crate::validate_assignment!(asgn) {
            Ok(PriorProcess { process, asgn })
        } else {
            asgn.validate()
                .emit_error()
                .map_err(BuildPriorProcessError::AssignmentError)?;
            Ok(PriorProcess { process, asgn })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1E-12;

    mod sb_slice {
        use super::*;

        #[test]
        fn should_return_input_weights_if_alpha_is_zero() {
            let mut rng = rand::thread_rng();
            let weights_in: Vec<f64> = vec![0.8, 0.2, 0.0];
            let weights_out =
                sb_slice_extend(weights_in.clone(), 1.0, 0.0, 0.2, &mut rng)
                    .unwrap();
            let good = weights_in
                .iter()
                .zip(weights_out.iter())
                .all(|(wi, wo)| (wi - wo).abs() < TOL);
            assert!(good);
        }

        #[test]
        fn should_return_error_for_zero_u_star() {
            let mut rng = rand::thread_rng();
            let weights_in: Vec<f64> = vec![0.8, 0.2];
            let u_star = 0.0;
            let res = sb_slice_extend(weights_in, 1.0, 0.0, u_star, &mut rng);
            assert!(res.is_err());
        }

        #[test]
        fn smoke() {
            let mut rng = rand::thread_rng();
            let weights_in: Vec<f64> = vec![0.8, 0.2];
            let u_star = 0.1;
            let res = sb_slice_extend(weights_in, 1.0, 0.0, u_star, &mut rng);
            assert!(res.is_ok());
        }
    }
}
