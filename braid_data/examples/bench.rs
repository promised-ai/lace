fn gen_parts<R, T, F>(
    n: usize,
    sparisty: f64,
    continuity: f64,
    gen_fn: F,
    mut rng: &mut R,
) -> (Vec<T>, Vec<bool>)
where
    R: rand::Rng,
    T: Copy + Default,
    F: Fn(&mut R) -> T,
{
    assert!(0.0 <= sparisty && sparisty < 1.0);
    assert!(0.0 <= continuity && continuity <= 1.0);

    let (n_slices, n_present) = if sparisty == 0.0 {
        (1, n)
    } else {
        let n_present =
            (((n as f64) * (1.0 - sparisty)).trunc() + 0.5) as usize;
        let sc = ((n_present as f64 * continuity).trunc() + 0.5) as usize;
        dbg!(n_present, sc);
        (n_present / sc.max(1), n_present)
    };

    let slice_size = n / n_slices;
    dbg!(slice_size, n_slices);

    let mut markers_placed: usize = 0;
    let mut slice_ix = 0;
    let mut rem = 0;

    let mut present = vec![false; n];
    while markers_placed < n_present {
        present[rem + slice_size * slice_ix] = true;

        slice_ix += 1;
        if slice_ix == n_slices {
            rem += 1;
            slice_ix = 0;
        }
        markers_placed += 1;
    }

    let data: Vec<T> = (0..n).map(|_| gen_fn(&mut rng)).collect();

    (data, present)
}

fn main() {
    let mut rng = rand::thread_rng();
    let (_, pr) = gen_parts(10, 0.5, 0.5, |&mut _r| 1.0, &mut rng);
    println!("{:?}", pr);
}
