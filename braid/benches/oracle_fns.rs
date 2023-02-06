use braid::examples::Example;
use braid::{
    Given, ImputeUncertaintyType, Oracle, OracleT, PredictUncertaintyType,
};
use braid_data::Datum;
use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

/// TODO: static states for for benchmarks
fn get_oracle() -> Oracle {
    Example::Animals.oracle().unwrap()
}

fn get_satellites_oracle() -> Oracle {
    Example::Satellites.oracle().unwrap()
}

fn bench_categorical_mi(c: &mut Criterion) {
    use braid::examples::satellites::Column;
    use braid::MiType;
    c.bench_function("oracle mi categorical", |b| {
        let oracle = get_oracle();
        b.iter(|| {
            let _mi = black_box(oracle.mi(
                Column::CountryOfOperator.into(),
                Column::Purpose.into(),
                1_000,
                MiType::UnNormed,
            ));
        })
    });
}

fn bench_continuous_mi(c: &mut Criterion) {
    use braid::examples::satellites::Column;
    use braid::MiType;
    c.bench_function("oracle mi continuous", |b| {
        let oracle = get_satellites_oracle();
        b.iter(|| {
            let _mi = black_box(oracle.mi(
                Column::ExpectedLifetime.into(),
                Column::PeriodMinutes.into(),
                1_000,
                MiType::UnNormed,
            ));
        })
    });
}

fn bench_catcon_mi(c: &mut Criterion) {
    use braid::examples::satellites::Column;
    use braid::MiType;
    c.bench_function("oracle mi categorical-continuous", |b| {
        let oracle = get_satellites_oracle();
        b.iter(|| {
            // Columns chosen so there is a about a 0.5 dependence probability
            let _mi = black_box(oracle.mi(
                Column::CountryOfOperator.into(),
                Column::ExpectedLifetime.into(),
                1_000,
                MiType::UnNormed,
            ));
        })
    });
}

fn bench_res(c: &mut Criterion) {
    c.bench_function("oracle ftype", |b| {
        let oracle = get_oracle();
        b.iter(|| {
            let _res = black_box(oracle.ftype(10));
        })
    });
}

fn bench_ress(c: &mut Criterion) {
    c.bench_function("oracle ftypes", |b| {
        let oracle = get_oracle();
        b.iter(|| {
            let _ress = black_box(oracle.ftypes());
        })
    });
}

fn bench_depprob(c: &mut Criterion) {
    c.bench_function("oracle depprob", |b| {
        let oracle = get_oracle();
        b.iter(|| {
            let _res = black_box(oracle.depprob(13, 10));
        })
    });
}

fn bench_rowsim(c: &mut Criterion) {
    c.bench_function("oracle rowsim", |b| {
        let oracle = get_oracle();
        b.iter(|| {
            let _res = black_box(oracle.rowsim(13, 10, None, false));
        })
    });
}

fn bench_novelty(c: &mut Criterion) {
    c.bench_function("oracle novelty", |b| {
        let oracle = get_oracle();
        b.iter(|| {
            let _res = black_box(oracle.novelty(13, None));
        })
    });
}

fn bench_cat_entropy(c: &mut Criterion) {
    c.bench_function("oracle categorical entropy", |b| {
        let oracle = get_oracle();
        b.iter(|| {
            let _res = black_box(oracle.entropy(&[1, 2, 3], 1000));
        })
    });
}

fn bench_predictor_search(c: &mut Criterion) {
    c.bench_function("oracle predictor search", |b| {
        let oracle = get_oracle();
        b.iter(|| {
            let _res = black_box(oracle.predictor_search(&[13], 2, 1000));
        })
    });
}

fn bench_info_prop(c: &mut Criterion) {
    c.bench_function("oracle info prop", |b| {
        let oracle = get_oracle();
        b.iter(|| {
            let _res = black_box(oracle.info_prop(&[13], &[2, 12], 1000));
        })
    });
}

fn bench_conditional_entropy(c: &mut Criterion) {
    c.bench_function("oracle conditional entropy", |b| {
        let oracle = get_oracle();
        b.iter(|| {
            let _res = black_box(oracle.conditional_entropy(13, &[1, 2], 1000));
        })
    });
}

fn bench_surprisal(c: &mut Criterion) {
    c.bench_function("oracle surprisal", |b| {
        let oracle = get_oracle();
        b.iter(|| {
            let x = black_box(Datum::Categorical(0));
            let _res = oracle.surprisal(&x, 13, 12, None);
        })
    });
}

fn bench_self_surprisal(c: &mut Criterion) {
    c.bench_function("oracle self surprisal", |b| {
        let oracle = get_oracle();
        b.iter(|| {
            let _res = black_box(oracle.self_surprisal(13, 12, None));
        })
    });
}

fn bench_datum(c: &mut Criterion) {
    c.bench_function("oracle datum", |b| {
        let oracle = get_oracle();
        b.iter(|| {
            let _res = black_box(oracle.datum(13, 12));
        })
    });
}

fn bench_logp(c: &mut Criterion) {
    c.bench_function("oracle logp", |b| {
        let given = Given::Conditions(vec![
            (0, Datum::Categorical(1)),
            (2, Datum::Categorical(0)),
        ]);
        let col_ixs = black_box(vec![3, 4]);
        let vals = vec![
            vec![Datum::Categorical(0), Datum::Categorical(0)],
            vec![Datum::Categorical(1), Datum::Categorical(1)],
        ];
        let oracle = get_oracle();
        b.iter(|| {
            let _res = oracle.logp(&col_ixs, &vals, &given, None);
        })
    });
}

fn bench_draw(c: &mut Criterion) {
    c.bench_function("oracle draw", |b| {
        let oracle = get_oracle();
        let mut rng = Xoshiro256Plus::seed_from_u64(1338);
        b.iter(|| {
            let _res = black_box(oracle.draw(13, 12, 100, &mut rng));
        })
    });
}

fn bench_simulate(c: &mut Criterion) {
    c.bench_function("oracle simulate", |b| {
        let given = Given::Conditions(vec![
            (0, Datum::Categorical(1)),
            (2, Datum::Categorical(0)),
        ]);
        let col_ixs = black_box(vec![3, 4]);
        let oracle = get_oracle();
        let mut rng = Xoshiro256Plus::seed_from_u64(1338);
        b.iter(|| {
            let _res = oracle.simulate(&col_ixs, &given, 100, None, &mut rng);
        })
    });
}

fn bench_impute(c: &mut Criterion) {
    c.bench_function("oracle impute", |b| {
        let oracle = get_oracle();
        let unc_type = ImputeUncertaintyType::JsDivergence;
        b.iter(|| {
            let _res = black_box(oracle.impute(13, 12, Some(unc_type)));
        })
    });
}

fn bench_predict(c: &mut Criterion) {
    c.bench_function("oracle predict", |b| {
        let given = Given::Conditions(vec![
            (0, Datum::Categorical(1)),
            (2, Datum::Categorical(0)),
        ]);
        let oracle = get_oracle();
        let unc_type = PredictUncertaintyType::JsDivergence;
        b.iter(|| {
            let _res =
                black_box(oracle.predict(13, &given, Some(unc_type), None));
        })
    });
}

fn bench_predict_continous(c: &mut Criterion) {
    c.bench_function("oracle predict continuous", |b| {
        let given = Given::Conditions(vec![(4, Datum::Categorical(3))]);
        let oracle = get_satellites_oracle();
        b.iter(|| {
            let _res = black_box(oracle.predict(8, &given, None, Nine));
        })
    });
}
criterion_group!(
    oracle_benches,
    bench_catcon_mi,
    bench_continuous_mi,
    bench_categorical_mi,
    bench_res,
    bench_ress,
    bench_depprob,
    bench_rowsim,
    bench_novelty,
    bench_cat_entropy,
    bench_predictor_search,
    bench_info_prop,
    bench_conditional_entropy,
    bench_surprisal,
    bench_self_surprisal,
    bench_datum,
    bench_logp,
    bench_simulate,
    bench_draw,
    bench_impute,
    bench_predict,
    bench_predict_continous,
);
criterion_main!(oracle_benches);
