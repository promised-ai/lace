/// Factorial, n!
///
/// # Example
///
/// ```
/// use lace::utils::numbers::factorial;
///
/// let fact = factorial(3);
/// assert_eq!(fact, 6);
/// ```
pub fn factorial(n: u64) -> u64 {
    if n < 2 {
        1
    } else {
        // (2..=n).fold(1, |acc, k| acc * k)
        (2..=n).product()
    }
}

/// Binomial coefficient, n choose k
///
/// # Example
///
/// ```
/// use lace::utils::numbers::binom;
///
/// let five_choose_two = binom(5, 2);
/// let five_choose_three = binom(5, 3);
/// let five_choose_four = binom(5, 4);
///
/// assert_eq!(five_choose_two, 10);
/// assert_eq!(five_choose_three, 10);
/// assert_eq!(five_choose_four, 5);
/// ```
pub fn binom(n: u64, k: u64) -> u64 {
    if k < 1 {
        1
    } else if k == 1 || n - k == 1 {
        n
    } else if n > 2 * k {
        // let numer = (n - k + 1..=n).fold(1, |acc, x| acc * x);
        let numer: u64 = (n - k + 1..=n).product();
        numer / factorial(k)
    } else {
        // let numer = (k + 1..=n).fold(1, |acc, x| acc * x);
        let numer: u64 = (k + 1..=n).product();
        numer / factorial(n - k)
    }
}

/// Stirling number of the [2nd
/// kind](https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind)
///
/// The number of ways to partition n items into k subsets
///
/// # Example
///
/// ```
/// use lace::utils::numbers::stirling;
///
/// assert_eq!( stirling(3, 2), 3);
/// assert_eq!( stirling(7, 2), 63);
/// assert_eq!( stirling(7, 6), 21);
/// assert_eq!( stirling(10, 3), 9330);
/// ```
#[allow(clippy::many_single_char_names)]
pub fn stirling(n: u64, k: u64) -> u64 {
    let sum: u64 = (0..=k).fold(0_i64, |acc, j| {
        let a = (-1_i64).pow((k - j) as u32);
        let b = binom(k, j) as i64;
        let c = (j as i64).pow(n as u32);
        acc + a * b * c
    }) as u64;
    sum / factorial(k)
}

/// The number of ways to partition n items into 1...n subsets
///
/// # Example
///
/// ```
/// use lace::utils::numbers::bell;
///
/// assert_eq!( bell(0), 1 );
/// assert_eq!( bell(1), 1 );
/// assert_eq!( bell(2), 2 );
/// assert_eq!( bell(3), 5 );
/// assert_eq!( bell(4), 15 );
/// assert_eq!( bell(5), 52 );
/// ```
pub fn bell(n: u64) -> u64 {
    (0..=n).fold(0_u64, |acc, k| acc + stirling(n, k))
}

/// The number of bi-partitions of an n-by-m (rows-by-columns) table
///
/// # Example
///
/// ```
/// use lace::utils::numbers::ccnum;
///
/// assert_eq!( ccnum(1, 1), 1 );
/// assert_eq!( ccnum(1, 2), 2 );
/// assert_eq!( ccnum(2, 1), 2 );
/// assert_eq!( ccnum(2, 2), 6 );
/// assert_eq!( ccnum(3, 3), 205 );
/// assert_eq!( ccnum(4, 3), 4065);
/// ```
pub fn ccnum(n: u64, m: u64) -> u64 {
    (0..=m).fold(0_u64, |acc, k| acc + stirling(m, k) * bell(n).pow(k as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let ans = 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11;
        assert_eq!(factorial(11), ans);
        assert_eq!(factorial(11), 39_916_800);
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
    fn stirling_nk() {
        assert_eq!(stirling(0, 0), 1);

        assert_eq!(stirling(1, 0), 0);
        assert_eq!(stirling(1, 1), 1);

        assert_eq!(stirling(10, 3), 9330);
        assert_eq!(stirling(10, 4), 34105);
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

    #[test]
    fn ccnum_1_1() {
        assert_eq!(ccnum(1, 1), 1);
    }

    #[test]
    fn ccnum_1_2() {
        assert_eq!(ccnum(1, 2), 2);
    }

    #[test]
    fn ccnum_2_1() {
        assert_eq!(ccnum(2, 1), 2);
    }

    #[test]
    fn ccnums() {
        assert_eq!(ccnum(2, 2), 6);
        assert_eq!(ccnum(3, 3), 205);
        assert_eq!(ccnum(4, 3), 4065);
    }
}
