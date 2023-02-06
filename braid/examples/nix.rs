use braid_stats::rv::dist::{Gamma, Gaussian, InvGamma, ScaledInvChiSquared};
use braid_stats::rv::traits::Rv;
use braid_utils::mean_var;

use plotly::layout::{Axis, GridPattern, Layout, LayoutGrid};
use plotly::{Histogram, Plot};

fn main() {
    let mut rng = rand::thread_rng();

    let mean: f64 = 18429.39;
    let sigma: f64 = 28502.115;
    // let mean: f64 = 0.0;
    // let sigma: f64 = 1.0;
    let var = sigma * sigma;
    let n: usize = 100_000;
    let logn = (n as f64).ln();
    let n_samples: usize = 100_000;

    let max_var = var * 100.0;
    let min_mean = mean - 10.0 * sigma;
    let max_mean = mean + 10.0 * sigma;

    let xs: Vec<f64> = Gaussian::new_unchecked(mean, sigma).sample(n, &mut rng);

    let (mx, vx) = mean_var(&xs);
    let sx = vx.sqrt();

    let pr_m = Gaussian::new(mx, sx).unwrap();
    let pr_k = Gamma::new(1.0, 1.0).unwrap();
    let pr_v = InvGamma::new(logn, logn).unwrap();
    let pr_s2 = InvGamma::new(logn, vx).unwrap();

    let mut means: Vec<f64> = Vec::with_capacity(n_samples);
    let mut vars: Vec<f64> = Vec::with_capacity(n_samples);
    let mut stddevs: Vec<f64> = Vec::with_capacity(n_samples);

    let mut n_rejected = 0;

    while means.len() < n_samples {
        let m: f64 = pr_m.draw(&mut rng);
        let k: f64 = pr_k.draw(&mut rng);
        let v: f64 = pr_v.draw(&mut rng);
        let s2: f64 = pr_s2.draw(&mut rng);

        let var: f64 = ScaledInvChiSquared::new(v, s2).unwrap().draw(&mut rng);

        let post_sigma = var.sqrt() / k.sqrt();
        let mu: f64 = Gaussian::new(m, post_sigma).unwrap().draw(&mut rng);

        if !(mean > min_mean && mean < max_mean && var < max_var) {
            n_rejected += 1;
            continue;
        }

        means.push(mu);
        vars.push(var);
        stddevs.push(var.sqrt());
    }

    println!(
        "{}% of samples rejected",
        (n_rejected as f64) / (n_samples as f64) * 100.0
    );

    let trace_xs = Histogram::new(xs).name("xs");
    let trace_means = Histogram::new(means)
        .name("mean")
        .x_axis("x2")
        .y_axis("y2")
        .n_bins_x(10000);
    let trace_vars = Histogram::new(vars)
        .name("var")
        .x_axis("x3")
        .y_axis("y3")
        .n_bins_x(1000);
    let trace_stddevs = Histogram::new(stddevs)
        .name("stddev")
        .x_axis("x4")
        .y_axis("y4")
        .n_bins_x(1000);

    let mut plot = Plot::new();

    plot.add_trace(trace_xs);
    plot.add_trace(trace_means);
    plot.add_trace(trace_vars);
    plot.add_trace(trace_stddevs);

    let layout = Layout::new()
        .grid(
            LayoutGrid::new()
                .rows(2)
                .columns(2)
                .pattern(GridPattern::Independent),
        )
        .x_axis(Axis::new().range(vec![min_mean, max_mean]))
        .x_axis2(Axis::new().range(vec![min_mean, max_mean]).anchor("y2"))
        .x_axis3(Axis::new().range(vec![0.0, max_var]).anchor("y3"))
        .x_axis4(Axis::new().range(vec![0.0, max_var.sqrt()]).anchor("y4"));

    plot.set_layout(layout);
    plot.show();
}
