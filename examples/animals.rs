use braid::examples::animals::Column;
use braid::examples::Example;
use std::convert::TryInto;

fn main() {
    let oracle = Example::Animals.oracle().unwrap();

    let predictors_swims: Vec<(Column, f64)> = oracle
        .predictor_search(&vec![Column::Swims.into()], 5, 10_000)
        .drain(..)
        .map(|(ix, info_prop)| (ix.try_into().unwrap(), info_prop))
        .collect();

    println!("{:#?}", predictors_swims);

    let predictors_domestic: Vec<(Column, f64)> = oracle
        .predictor_search(&vec![Column::Domestic.into()], 5, 10_000)
        .drain(..)
        .map(|(ix, info_prop)| (ix.try_into().unwrap(), info_prop))
        .collect();

    println!("{:#?}", predictors_domestic);
}
