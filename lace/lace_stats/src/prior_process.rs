use crate::rv::dist::{Beta, Gamma};
use lace_consts::rv::{misc::pflip, traits::Rv};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::assignment::Assignment;

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
        // TODO: remove clones
        *self = Self::from_prior(self.prior.clone(), rng);
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
        unimplemented!()
    }

    fn update_params<R: Rng>(&mut self, asgn: &Assignment, rng: &mut R) -> f64 {
        unimplemented!()
    }

    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        // TODO: remove clones
        *self = Self::from_prior(
            self.prior_alpha.clone(),
            self.prior_d.clone(),
            rng,
        );
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
    let n_cats = weights.len() as f64;

    let mut beta = Beta::new(1.0 + d, alpha).unwrap();

    let mut iters: u16 = 0;
    loop {
        if d > 0.0 {
            let b = (iters as f64) + n_cats + 1.0;
            beta.set_beta(alpha + b * d).unwrap();
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
            return Err(TheStickIsDust(MAX_STICK_BREAKING_ITERS));
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
