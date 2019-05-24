use braid_codebook::codebook::{Codebook, ColType};
use braid_stats::prior::{Csd, Ng};
use rusqlite::types::{FromSql, ToSql};
use rusqlite::Connection;

use crate::cc::{ColModel, Column, DataContainer};
use crate::data::traits::SqlDefault;

// See https://users.rust-lang.org/t/sql-parameter-values/20469
const NO_ARGS: &'static [&'static ToSql] = &[];

/// Use a `cc::Codebook` to convert SQL database columns into column models
pub fn read_cols(conn: &Connection, codebook: &Codebook) -> Vec<ColModel> {
    let mut rng = rand::thread_rng();
    let table = &codebook.table_name;

    codebook
        .col_metadata
        .iter()
        .map(|(name, colmd)| match colmd.coltype {
            ColType::Continuous { ref hyper } => {
                let data = sql_to_container(name, &table, &conn);
                let prior = if hyper.is_some() {
                    let hyper_cpy = hyper.clone().unwrap();
                    Ng::from_hyper(hyper_cpy, &mut rng)
                } else {
                    Ng::from_data(&data.data, &mut rng)
                };
                let column = Column::new(colmd.id, data, prior);
                ColModel::Continuous(column)
            }
            ColType::Categorical { k, ref hyper, .. } => {
                let data = sql_to_container(name, &table, &conn);
                let prior = if hyper.is_some() {
                    let hyper_cpy = hyper.clone().unwrap();
                    Csd::from_hyper(k, hyper_cpy, &mut rng)
                } else {
                    Csd::vague(k, &mut rng)
                };
                let column = Column::new(colmd.id, data, prior);
                ColModel::Categorical(column)
            }
            ColType::Binary { .. } => {
                unimplemented!();
            }
        })
        .collect()
}

/// Read a SQL column into a `cc::DataContainer`.
fn sql_to_container<T>(col: &str, table: &str, conn: &Connection) -> DataContainer<T>
where
    T: Clone + FromSql + SqlDefault,
{
    // FIXME: Dangerous!!!
    let query = format!("SELECT {} from {} ORDER BY id ASC;", col, table);
    let mut stmnt = conn.prepare(query.as_str()).unwrap();
    let data_iter = stmnt
        .query_map(NO_ARGS, |row| match row.get(0) {
            Ok(x) => Ok((x, true)),
            Err(_) => Ok((T::sql_default(), false)),
        })
        .unwrap();

    // TODO:preallocate
    let mut data = Vec::new();
    let mut present = Vec::new();
    data_iter.for_each(|val| {
        let (x, pr) = val.unwrap();
        data.push(x);
        present.push(pr);
    });

    DataContainer { data, present }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use braid_codebook::codebook::{ColMetadata, ColType, SpecType};
    use maplit::btreemap;

    fn multi_type_data() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE data (
                      id  INTEGER PRIMARY KEY,
                      x   REAL,
                      y   INTEGER
                      )",
            NO_ARGS,
        )
        .unwrap();
        conn.execute(
            "INSERT INTO data (id, x, y)
                      VALUES (0, 1.2, NULL)",
            NO_ARGS,
        )
        .unwrap();
        conn.execute(
            "INSERT INTO data (id, x, y)
                      VALUES (1, 2.3, 2)",
            NO_ARGS,
        )
        .unwrap();
        conn.execute(
            "INSERT INTO data (id, x, y)
                      VALUES (2, 3.4, 1)",
            NO_ARGS,
        )
        .unwrap();
        conn.execute(
            "INSERT INTO data (id, x, y)
                      VALUES (3, 4.5, 0)",
            NO_ARGS,
        )
        .unwrap();
        conn.execute(
            "INSERT INTO data (id, x, y)
                      VALUES (4, NULL, 0)",
            NO_ARGS,
        )
        .unwrap();
        conn
    }

    fn single_real_column_no_missing() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE data (
                      id  INTEGER PRIMARY KEY,
                      x   REAL
                      )",
            NO_ARGS,
        )
        .unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (0, 1.2)", NO_ARGS)
            .unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (1, 2.3)", NO_ARGS)
            .unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (2, 3.4)", NO_ARGS)
            .unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (3, 4.5)", NO_ARGS)
            .unwrap();
        conn
    }

    fn single_int_column_no_missing() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE data (
                      id  INTEGER PRIMARY KEY,
                      x   INTEGER
                      )",
            NO_ARGS,
        )
        .unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (0, 1)", NO_ARGS)
            .unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (1, 2)", NO_ARGS)
            .unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (2, 3)", NO_ARGS)
            .unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (3, 4)", NO_ARGS)
            .unwrap();
        conn
    }

    #[test]
    fn read_continuous_data_with_no_missing() {
        let conn = single_real_column_no_missing();

        let data: DataContainer<f64> = sql_to_container(&"x", &"data", &conn);

        assert_eq!(data.len(), 4);
        assert_relative_eq!(data[0], 1.2, epsilon = 10E-10);
        assert_relative_eq!(data[1], 2.3, epsilon = 10E-10);
        assert_relative_eq!(data[2], 3.4, epsilon = 10E-10);
        assert_relative_eq!(data[3], 4.5, epsilon = 10E-10);
    }

    #[test]
    fn continuous_data_should_be_ordered_by_id() {
        let conn = single_real_column_no_missing();

        // Add data out of order (by id)
        conn.execute("INSERT INTO data (id, x) VALUES (5, 5.5)", NO_ARGS)
            .unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (4, 4.4)", NO_ARGS)
            .unwrap();

        let data: DataContainer<f64> = sql_to_container(&"x", &"data", &conn);

        assert_eq!(data.len(), 6);
        assert_relative_eq!(data[0], 1.2, epsilon = 10E-10);
        assert_relative_eq!(data[1], 2.3, epsilon = 10E-10);
        assert_relative_eq!(data[2], 3.4, epsilon = 10E-10);
        assert_relative_eq!(data[3], 4.5, epsilon = 10E-10);
        assert_relative_eq!(data[4], 4.4, epsilon = 10E-10);
        assert_relative_eq!(data[5], 5.5, epsilon = 10E-10);
    }

    #[test]
    fn read_categorical_data_with_no_missing() {
        let conn = single_int_column_no_missing();

        let data: DataContainer<u8> = sql_to_container(&"x", &"data", &conn);

        assert_eq!(data.len(), 4);
        assert_eq!(data[0], 1);
        assert_eq!(data[1], 2);
        assert_eq!(data[2], 3);
        assert_eq!(data[3], 4);
    }

    #[test]
    fn continuous_data_should_read_nulls_as_default() {
        let conn = single_real_column_no_missing();

        // Add data out of order (by id)
        conn.execute("INSERT INTO data (id, x) VALUES (4, NULL)", NO_ARGS)
            .unwrap();

        let data: DataContainer<f64> = sql_to_container(&"x", &"data", &conn);

        assert_eq!(data.len(), 5);
        assert_relative_eq!(data[0], 1.2, epsilon = 10E-10);
        assert_relative_eq!(data[1], 2.3, epsilon = 10E-10);
        assert_relative_eq!(data[2], 3.4, epsilon = 10E-10);
        assert_relative_eq!(data[3], 4.5, epsilon = 10E-10);
        assert_relative_eq!(data[4], 0.0, epsilon = 10E-10);

        assert!(data.present[0]);
        assert!(data.present[1]);
        assert!(data.present[2]);
        assert!(data.present[3]);
        assert!(!data.present[4]);
    }

    #[test]
    fn read_db_should_return_correct_number_of_columns() {
        let conn = multi_type_data();
        let codebook = Codebook {
            view_alpha_prior: None,
            state_alpha_prior: None,
            comments: None,
            row_names: None,
            table_name: String::from("data"),
            col_metadata: btreemap!(
                String::from("x") => ColMetadata {
                    id: 0,
                    spec_type: SpecType::Other,
                    name: String::from("x"),
                    coltype: ColType::Continuous { hyper: None },
                    notes: None,
                },
                String::from("y") => ColMetadata {
                    id: 1,
                    spec_type: SpecType::Other,
                    name: String::from("y"),
                    coltype: ColType::Categorical {
                        k: 3,
                        hyper: None,
                        value_map: None,
                    },
                    notes: None,
                },
            ),
        };
        assert!(codebook.validate_ids().is_ok());

        let col_models = read_cols(&conn, &codebook);
        assert_eq!(col_models.len(), 2);
        assert!(col_models[0].is_continuous());
        assert!(col_models[1].is_categorical());
    }
}
