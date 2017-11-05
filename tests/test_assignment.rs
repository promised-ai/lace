extern crate rand;
extern crate braid;

use rand::XorShiftRng;
use braid::cc::assignment::Assignment;


#[test]
fn drawn_assignment_should_have_valid_partition() {
    let n: usize = 50;
    let alpha: f64 = 1.0;
    let mut rng = XorShiftRng::new_unseeded();

    // do the test 100 times because it's random
    for _ in 0..100 {
        let asgn = Assignment::draw(n, alpha, &mut rng);
        let max_ix = *asgn.asgn.iter().max().unwrap();
        let min_ix = *asgn.asgn.iter().min().unwrap();

        assert_eq!(asgn.counts.len(), asgn.ncats);
        assert_eq!(asgn.counts.len(), max_ix + 1);
        assert_eq!(min_ix, 0);

        for (k, &count) in asgn.counts.iter().enumerate() {
            let k_count = asgn.asgn.iter().fold(0, |acc, &z| {
                if z == k {acc + 1} else {acc}
            });
            assert_eq!(k_count, count);
        }
    }
}


#[test]
fn flat_partition_validation() {
    let n: usize = 50;
    let alpha: f64 = 1.0;

    let asgn = Assignment::flat(n, alpha);

    assert_eq!(asgn.ncats, 1);
    assert_eq!(asgn.counts.len(), 1);
    assert_eq!(asgn.counts[0], n);
    assert!(asgn.asgn.iter().all(|&z| z == 0));
}
