use rv::dist::{Gamma, InvGamma};
use rv::traits::{Mean, Rv, Variance};

use braid_stats::mat::{Matrix2x2, Vector2};
use braid_stats::mh::mh_symrw_adaptive_mv;
use braid_stats::prior::pg::PgHyper;

fn run() {
    let mut rng = rand::thread_rng();
    let rates: Vec<f64> = Gamma::new_unchecked(2.0, 2.0).sample(10, &mut rng);

    let hyper = PgHyper {
        pr_shape: InvGamma::new(3.0, 4.0).unwrap(),
        pr_rate: InvGamma::new(4.0, 4.0).unwrap(),
    };

    let score_fn = |shape_rate: &[f64]| {
        let shape = shape_rate[0];
        let rate = shape_rate[1];
        let gamma = Gamma::new(shape, rate).unwrap();
        let loglike = rates.iter().map(|rate| gamma.ln_f(rate)).sum::<f64>();
        let prior = hyper.pr_rate.ln_f(&rate) + hyper.pr_shape.ln_f(&shape);
        loglike + prior
    };

    let n_steps: usize = 1_000;
    let mut x = vec![1.0, 1.0];

    let mut shapes = Vec::with_capacity(n_steps);
    let mut rates = Vec::with_capacity(n_steps);
    let mut ln_scores = Vec::with_capacity(n_steps);

    for _ in 0..1000 {
        let mh_result = mh_symrw_adaptive_mv(
            Vector2([x[0], x[1]]),
            Vector2([
                hyper.pr_shape.mean().unwrap(),
                hyper.pr_rate.mean().unwrap(),
            ]),
            Matrix2x2::from_diag([
                hyper.pr_shape.variance().unwrap(),
                hyper.pr_rate.variance().unwrap(),
            ]),
            50,
            score_fn,
            &vec![(0.0, std::f64::INFINITY), (0.0, std::f64::INFINITY)],
            &mut rng,
        );
        x = mh_result.x;

        shapes.push(x[0]);
        rates.push(x[1]);
        ln_scores.push(mh_result.score_x);
    }
}

fn main() {
    run()
}
