extern crate rand;
extern crate rusqlite;

use self::rusqlite::types::FromSql;
use self::rusqlite::Connection;

use cc::codebook::ColMetadata;
use cc::{Codebook, ColModel, Column, DataContainer};
use data::traits::SqlDefault;
use dist::prior::{CatSymDirichlet, NormalInverseGamma};

/// Use a `cc::Codebook` to convert SQL database columns into column models
pub fn read_cols(conn: &Connection, codebook: &Codebook) -> Vec<ColModel> {
    let mut rng = rand::thread_rng();
    let table = &codebook.table_name;
    let colmds = codebook.zip_col_metadata();

    colmds
        .iter()
        .map(|(id, name, colmd)| match colmd {
            &ColMetadata::Continuous { ref hyper } => {
                let data = sql_to_container(&name, &table, &conn);
                let prior = if hyper.is_some() {
                    let hyper_cpy = hyper.clone().unwrap();
                    NormalInverseGamma::from_hyper(hyper_cpy, &mut rng)
                } else {
                    NormalInverseGamma::from_data(&data.data, &mut rng)
                };
                let column = Column::new(*id, data, prior);
                ColModel::Continuous(column)
            }
            &ColMetadata::Categorical {
                k, ref hyper, ..
            } => {
                let data = sql_to_container(&name, &table, &conn);
                let prior = if hyper.is_some() {
                    let hyper_cpy = hyper.clone().unwrap();
                    CatSymDirichlet::from_hyper(k, hyper_cpy, &mut rng)
                } else {
                    CatSymDirichlet::vague(k, &mut rng)
                };
                let column = Column::new(*id, data, prior);
                ColModel::Categorical(column)
            }
            &ColMetadata::Binary { .. } => {
                unimplemented!();
            }
        })
        .collect()
}

/// Read a SQL column into a `cc::DataContainer`.
fn sql_to_container<T>(
    col: &str,
    table: &str,
    conn: &Connection,
) -> DataContainer<T>
where
    T: Clone + FromSql + SqlDefault,
{
    // FIXME: Dangerous!!!
    let query = format!("SELECT {} from {} ORDER BY id ASC;", col, table);
    let mut stmnt = conn.prepare(query.as_str()).unwrap();
    let data_iter = stmnt
        .query_map(&[], |row| match row.get_checked(0) {
            Ok(x) => (x, true),
            Err(_) => (T::sql_default(), false),
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

    DataContainer {
        data: data,
        present: present,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cc::codebook::MetaData;
    use cc::SpecType;

    fn multi_type_data() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE data (
                      id  INTEGER PRIMARY KEY,
                      x   REAL,
                      y   INTEGER
                      )",
            &[],
        ).unwrap();
        conn.execute(
            "INSERT INTO data (id, x, y)
                      VALUES (0, 1.2, NULL)",
            &[],
        ).unwrap();
        conn.execute(
            "INSERT INTO data (id, x, y)
                      VALUES (1, 2.3, 2)",
            &[],
        ).unwrap();
        conn.execute(
            "INSERT INTO data (id, x, y)
                      VALUES (2, 3.4, 1)",
            &[],
        ).unwrap();
        conn.execute(
            "INSERT INTO data (id, x, y)
                      VALUES (3, 4.5, 0)",
            &[],
        ).unwrap();
        conn.execute(
            "INSERT INTO data (id, x, y)
                      VALUES (4, NULL, 0)",
            &[],
        ).unwrap();
        conn
    }

    fn single_real_column_no_missing() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE data (
                      id  INTEGER PRIMARY KEY,
                      x   REAL
                      )",
            &[],
        ).unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (0, 1.2)", &[])
            .unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (1, 2.3)", &[])
            .unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (2, 3.4)", &[])
            .unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (3, 4.5)", &[])
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
            &[],
        ).unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (0, 1)", &[])
            .unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (1, 2)", &[])
            .unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (2, 3)", &[])
            .unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (3, 4)", &[])
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
        conn.execute("INSERT INTO data (id, x) VALUES (5, 5.5)", &[])
            .unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (4, 4.4)", &[])
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
        conn.execute("INSERT INTO data (id, x) VALUES (4, NULL)", &[])
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
            comments: None,
            row_names: None,
            table_name: String::from("data"),
            metadata: vec![
                MetaData::Column {
                    id: 0,
                    spec_type: SpecType::Other,
                    name: String::from("x"),
                    colmd: ColMetadata::Continuous { hyper: None },
                },
                MetaData::Column {
                    id: 1,
                    spec_type: SpecType::Other,
                    name: String::from("y"),
                    colmd: ColMetadata::Categorical {
                        k: 3,
                        hyper: None,
                        value_map: None,
                    },
                },
                MetaData::StateAlpha { alpha: 1.0 },
                MetaData::ViewAlpha { alpha: 1.0 },
            ],
        };
        assert!(codebook.validate_ids().is_ok());

        let col_models = read_cols(&conn, &codebook);
        assert_eq!(col_models.len(), 2);
        assert!(col_models[0].is_continuous());
        assert!(col_models[1].is_categorical());
    }
}