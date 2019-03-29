extern crate braid_utils;

use braid_utils::misc::n_unique;

pub fn is_categorical(col: &Vec<f64>, cutoff: u8) -> bool {
    // drop nan
    let xs: Vec<f64> =
        col.iter().filter(|x| x.is_finite()).map(|x| *x).collect();
    let all_ints = xs.iter().all(|&x| x.round() == x);
    if !all_ints {
        false
    } else {
        n_unique(&xs, Some(cutoff as usize)) <= (cutoff as usize)
    }
}
