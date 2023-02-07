mod enum_test;

use enum_test::*;

mod partition {
    use super::*;
    use lace_cc::misc::crp_draw;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn empty_partition() {
        let mut rng = StdRng::seed_from_u64(0xABCD);
        let draw = crp_draw(0, 1.0, &mut rng);
        let empty: Vec<usize> = vec![];
        assert_eq!(draw.asgn, empty);
        assert_eq!(draw.counts, empty);
        assert_eq!(draw.n_cats, 0);
    }

    #[test]
    fn single_element_partition() {
        let mut rng = StdRng::seed_from_u64(0xABCD);
        let draw = crp_draw(1, 1.0, &mut rng);
        let asgn: Vec<usize> = vec![0];
        let counts: Vec<usize> = vec![1];
        assert_eq!(draw.asgn, asgn);
        assert_eq!(draw.counts, counts);
        assert_eq!(draw.n_cats, 1);
    }

    #[test]
    fn two_element_partition() {
        let mut rng = StdRng::seed_from_u64(0xABCD);
        let draw = crp_draw(2, 1E6, &mut rng);
        let asgn: Vec<usize> = vec![0, 1];
        let counts: Vec<usize> = vec![1, 1];
        assert_eq!(draw.asgn, asgn);
        assert_eq!(draw.counts, counts);
        assert_eq!(draw.n_cats, 2);
    }

    #[test]
    fn partition_iterator_creates_right_number_of_partitions() {
        // https://en.wikipedia.org/wiki/Bell_number
        let bell_nums: Vec<(usize, u64)> =
            vec![(1, 1), (2, 2), (3, 5), (4, 15), (5, 52), (6, 203)];

        for (n, bell) in bell_nums {
            let mut count: u64 = 0;
            Partition::new(n).for_each(|_| count += 1);
            assert_eq!(count, bell);
        }
    }
}

mod enumeration {
    use super::*;

    #[test]
    fn test_partitiion_to_ix_on_binary() {
        assert_eq!(partition_to_ix(&vec![0, 0]), 0);
        assert_eq!(partition_to_ix(&vec![1, 0]), 1);
        assert_eq!(partition_to_ix(&vec![0, 1]), 2);
        assert_eq!(partition_to_ix(&vec![1, 1]), 3);
    }

    #[test]
    fn test_partitiion_to_ix_on_trinary() {
        assert_eq!(partition_to_ix(&vec![0, 0, 0]), 0);
        assert_eq!(partition_to_ix(&vec![1, 0, 0]), 1);
        assert_eq!(partition_to_ix(&vec![2, 0, 0]), 2);
        assert_eq!(partition_to_ix(&vec![0, 1, 0]), 3);
        assert_eq!(partition_to_ix(&vec![1, 1, 0]), 4);
        assert_eq!(partition_to_ix(&vec![2, 1, 0]), 5);
        assert_eq!(partition_to_ix(&vec![0, 2, 0]), 6);
    }

    #[test]
    fn normalize_assignment_one_partition() {
        let z: Vec<usize> = vec![0, 0, 0, 0];
        assert_eq!(normalize_assignment(z.clone()), z);
    }

    #[test]
    fn normalize_assignment_should_not_change_normalize_assignment() {
        let z: Vec<usize> = vec![0, 1, 2, 1];
        assert_eq!(normalize_assignment(z.clone()), z);
    }

    #[test]
    fn normalize_assignment_should_fix_assignment_1() {
        let target: Vec<usize> = vec![0, 1, 2, 1];
        let unnormed: Vec<usize> = vec![1, 0, 2, 0];
        assert_eq!(normalize_assignment(unnormed.clone()), target);
    }

    #[test]
    fn normalize_assignment_should_fix_assignment_2() {
        let target: Vec<usize> = vec![0, 0, 1, 2];
        let unnormed: Vec<usize> = vec![0, 0, 2, 1];
        assert_eq!(normalize_assignment(unnormed.clone()), target);
    }

    #[test]
    fn normalize_assignment_should_fix_assignment_3() {
        let target: Vec<usize> = vec![0, 1, 1, 2, 1];
        let unnormed: Vec<usize> = vec![2, 1, 1, 0, 1];
        assert_eq!(normalize_assignment(unnormed.clone()), target);
    }
}
