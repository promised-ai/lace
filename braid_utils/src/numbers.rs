/// Factorial, n!
pub fn factorial(n: u64) -> u64 {
    if n < 2 {
        1
    } else {
        // (2..=n).fold(1, |acc, k| acc * k)
        (2..=n).product()
    }
}

/// Binomial coefficient, n choose k
pub fn binom(n: u64, k: u64) -> u64 {
    if k < 1 {
        1
    } else if k == 1 || n - k == 1 {
        n
    } else if n - k > k {
        // let numer = (n - k + 1..=n).fold(1, |acc, x| acc * x);
        let numer: u64 = (n - k + 1..=n).product();
        numer / factorial(k)
    } else {
        // let numer = (k + 1..=n).fold(1, |acc, x| acc * x);
        let numer: u64 = (k + 1..=n).product();
        numer / factorial(n - k)
    }
}

/// Sterling number of the 2nd kind
///
/// The number of ways to partition n items into k subsets
pub fn sterling(n: u64, k: u64) -> u64 {
    let sum: u64 = (0..=k).fold(0_i64, |acc, j| {
        let a = (-1_i64).pow((k - j) as u32);
        let b = binom(k, j) as i64;
        let c = (j as i64).pow(n as u32);
        acc + a * b * c
    }) as u64;
    sum / factorial(k)
}

/// The number of ways to partition n items into 1...n subsets
pub fn bell(n: u64) -> u64 {
    (0..=n).fold(0_u64, |acc, k| acc + sterling(n, k))
}

/// The number of bi-partitions of an n-by-m (rows-by-columns) table
pub fn ccnum(n: u64, m: u64) -> u64 {
    (0..=m).fold(0_u64, |acc, k| acc + sterling(m, k) * bell(n).pow(k as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    // FIXME: test factorial

    #[test]
    fn factorial_0_should_be_1() {
        assert_eq!(factorial(0), 1);
    }

    #[test]
    fn factorial_1_should_be_1() {
        assert_eq!(factorial(1), 1);
    }

    #[test]
    fn factorial_2_should_be_2() {
        assert_eq!(factorial(2), 2);
    }

    #[test]
    fn factorial_11() {
        let ans = 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11;
        assert_eq!(factorial(11), ans);
        assert_eq!(factorial(11), 39916800);
    }

    #[test]
    fn binom_nk() {
        assert_eq!(binom(5, 0), 1);
        assert_eq!(binom(5, 1), 5);
        assert_eq!(binom(5, 2), 10);
        assert_eq!(binom(5, 3), 10);
        assert_eq!(binom(5, 4), 5);
        assert_eq!(binom(5, 1), 5);

        assert_eq!(binom(10, 6), 210);
        assert_eq!(binom(10, 4), 210);
    }

    #[test]
    fn sterling_nk() {
        assert_eq!(sterling(0, 0), 1);

        assert_eq!(sterling(1, 0), 0);
        assert_eq!(sterling(1, 1), 1);

        assert_eq!(sterling(10, 3), 9330);
        assert_eq!(sterling(10, 4), 34105);
    }

    #[test]
    fn bell_n() {
        assert_eq!(bell(0), 1);
        assert_eq!(bell(1), 1);
        assert_eq!(bell(2), 2);
        assert_eq!(bell(3), 5);
        assert_eq!(bell(4), 15);
        assert_eq!(bell(5), 52);
    }

    // FIXME: test ccnum
}
