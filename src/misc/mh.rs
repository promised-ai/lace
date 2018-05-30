extern crate rand;

use self::rand::distributions::Uniform;
use self::rand::Rng;
use std::f64;

pub fn mh_prior<T, F, D, R: Rng>(
    loglike: F,
    prior_draw: D,
    n_iter: usize,
    mut rng: &mut R,
) -> T
where
    F: Fn(&T) -> f64,
    D: Fn(&mut R) -> T,
{
    let u = Uniform::new(0.0, 1.0);
    let x = prior_draw(&mut rng);
    let fx = loglike(&x);
    (0..n_iter)
        .fold((x, fx), |(x, fx), _| {
            let y = prior_draw(&mut rng);
            let fy = loglike(&y);
            let r: f64 = rng.sample(u);
            if r.ln() < fy - fx {
                (y, fy)
            } else {
                (x, fx)
            }
        })
        .0
}

// TODO: Random Walk

#[cfg(test)]
mod tests {
    use self::rand::distributions::Normal;
    use super::*;

    #[test]
    fn test_mh_prior_uniform() {
        let loglike = |_x: &f64| 0.0;
        fn prior_draw<R: Rng>(r: &mut R) -> f64 {
            let u = Uniform::new(0.0, 1.0);
            r.sample(u)
        }

        let mut rng = rand::thread_rng();
        let x = mh_prior(loglike, prior_draw, 25, &mut rng);

        assert!(x <= 1.0);
        assert!(x >= 0.0);
    }

    #[test]
    fn test_mh_prior_normal() {
        let loglike = |_x: &f64| 0.0;
        fn prior_draw<R: Rng>(r: &mut R) -> f64 {
            let norm = Normal::new(0.0, 1.0);
            r.sample(norm)
        }

        let mut rng = rand::thread_rng();

        // Smoke test. just make sure nothing blows up
        let _x = mh_prior(loglike, prior_draw, 25, &mut rng);
    }
}
