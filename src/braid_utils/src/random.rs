extern crate rand;

/// Choose two distinct random numbers in [0, ..., n-1]
pub fn choose2ixs<R: rand::Rng>(n: usize, rng: &mut R) -> (usize, usize) {
    if n < 2 {
        panic!("n must be 2 or greater")
    } else if n == 2 {
        (0, 1)
    } else {
        let i: usize = rng.gen_range(0, n);
        loop {
            let j: usize = rng.gen_range(0, n);
            if j != i {
                return (i, j);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // FIXME
}
