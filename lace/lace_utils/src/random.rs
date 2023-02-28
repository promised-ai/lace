use std::cmp::Ordering;

/// Choose two distinct random numbers in [0, ..., n-1]
pub fn choose2ixs<R: rand::Rng>(n: usize, rng: &mut R) -> (usize, usize) {
    match n.cmp(&2) {
        Ordering::Greater => {
            let i: usize = rng.gen_range(0..n);
            loop {
                let j: usize = rng.gen_range(0..n);
                if j != i {
                    return (i, j);
                }
            }
        }
        Ordering::Equal => (0, 1),
        Ordering::Less => panic!("n must be 2 or greater"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose2ixs_should_return_different_ixs() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let (a, b) = choose2ixs(10, &mut rng);
            assert_ne!(a, b);
        }
    }
}
