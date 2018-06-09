pub struct EmpiricalCdf {
    xs: Vec<f64>,
    fx: Vec<f64>,
}

impl EmpiricalCdf {
    pub fn new(samples: &[f64]) -> Self {
        let mut xs = Vec::from(samples);
        xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let n = xs.len() as f64;
        let fx: Vec<f64> = (0..xs.len()).map(|i| i as f64 / n).collect();

        EmpiricalCdf { xs: xs, fx: fx } 
    }

    pub fn f(&self, values: &[f64]) -> Vec<f64> {
        values.iter().map(|value| {
            self.xs
                .iter()
                .enumerate()
                .find(|(_, x)| *x > value)
                .and_then(|(ix, _)| Some(self.fx[ix]))
                .unwrap_or(1.0)
        }).collect()
    }

    pub fn pp(&self, other: &Self) -> (Vec<f64>, Vec<f64>) {
        let mut xys = self.xs.clone();
        xys.append(&mut other.xs.clone());
        xys.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        (self.f(&xys), other.f(&xys))
    }
    
    /// Area between CDF-CDF line
    pub fn auc(&self, other: &Self) -> f64 {
        let (fxs, fys) = self.pp(other);
        let diff: Vec<f64> = fxs
            .iter()
            .zip(fys.iter())
            .map(|(fx, fy)| (fx - fy).abs())
            .collect();

        let mut q = 0.0;
        for i in 1..fxs.len() {
            let step = fxs[i] - fxs[i-1];
            let trap = diff[i] + diff[i-1];
            q += step * trap
        }
        q / 2.0
    }
}
