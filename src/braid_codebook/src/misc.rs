use std::mem::swap;
use std::str::FromStr;

// TODO: move misc funcs into their own crate
pub fn minmax<T: PartialOrd + Clone>(xs: &[T]) -> (T, T) {
    if xs.is_empty() {
        panic!("Empty slice");
    }

    if xs.len() == 1 {
        return (xs[0].clone(), xs[0].clone());
    }

    let mut min = &xs[0];
    let mut max = &xs[1];

    if min > max {
        swap(&mut min, &mut max);
    }

    for i in 2..xs.len() {
        if xs[i] > *max {
            max = &xs[i];
        } else if xs[i] < *min {
            min = &xs[i];
        }
    }

    (min.clone(), max.clone())
}

pub fn transpose(mat_in: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let nrows = mat_in.len();
    let ncols = mat_in[0].len();
    let mut mat_out: Vec<Vec<f64>> = vec![vec![0.0; nrows]; ncols];

    for (i, row) in mat_in.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            mat_out[j][i] = x;
        }
    }

    mat_out
}

pub fn parse_result<T: FromStr>(res: &str) -> Option<T> {
    // For csv, empty cells are considered missing regardless of type
    if res.is_empty() {
        None
    } else {
        match res.parse::<T>() {
            Ok(x) => Some(x),
            Err(_) => panic!("Could not parse \"{}\"", res),
        }
    }
}

fn n_unique(xs: &Vec<f64>, cutoff_opt: Option<usize>) -> usize {
    let mut unique: Vec<f64> = vec![xs[0]];
    match cutoff_opt {
        Some(cutoff) => {
            for x in xs.iter().skip(1) {
                if !unique.iter().any(|y| y == x) {
                    unique.push(*x);
                }
                if unique.len() > cutoff {
                    return unique.len();
                }
            }
        }
        None => {
            for x in xs.iter().skip(1) {
                if !unique.iter().any(|y| y == x) {
                    unique.push(*x);
                }
            }
        }
    }
    unique.len()
}

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
