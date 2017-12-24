extern crate rusqlite;

use std::path::Path;

use self::rusqlite::Connection;

use cc::{Codebook, ColModel, DataContainer, Column};
use cc::codebook::ColMetadata;
use dist::SymmetricDirichlet;
use dist::prior::{NormalInverseGamma};



pub fn col_models(path: &Path, codebook: Codebook) -> Vec<ColModel> {
    let table = &codebook.table_name;
    let conn = Connection::open(path).unwrap();
    let colmds = codebook.zip_col_metadata();

    colmds.iter().map(|(id, name, colmd)| {
        match colmd {
            &ColMetadata::Continuous {m, r, s, v} => {
                let data = sel_continuous_data(&name, &table, &conn);
                let prior = NormalInverseGamma::new(m, r, s, v);
                let column = Column::new(*id, data, prior);
                ColModel::Continuous(column)
            },
            &ColMetadata::Categorical {alpha, k} => {
                let data = sel_categorical_data(&name, &table, &conn);
                let prior = SymmetricDirichlet::new(alpha, k);
                let column = Column::new(*id, data, prior);
                ColModel::Categorical(column)
            },
            &ColMetadata::Binary {a, b} => {
                unimplemented!();
            },
        }
    }).collect()
}


fn sel_continuous_data(col: &str, table: &str, conn: &Connection)
    -> DataContainer<f64>
{
    // FIXME: Dangerous!!!
    let query = format!("SELECT {} from {} ORDER BY id ASC;", col, table);
    let mut stmnt = conn.prepare(query.as_str()).unwrap();
    let data = stmnt
        .query_map(&[], |r| r.get(0))
        .unwrap()
        .map(|val| val.unwrap())
        .collect();

    DataContainer::new(data)
}


fn sel_categorical_data(col: &str, table: &str, conn: &Connection)
    -> DataContainer<u8>
{
    // FIXME: Dangerous!!!
    let query = format!("SELECT {} from {} ORDER BY id ASC;", col, table);
    let mut stmnt = conn.prepare(query.as_str()).unwrap();
    let data: Vec<u8> = stmnt
        .query_map(&[], |r| r.get(0))
        .unwrap()
        .map(|val| val.unwrap())
        .collect();
    DataContainer::new(data)
}



#[cfg(test)]
mod tests {
    use super::*;

    fn single_real_column_no_missing() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute("CREATE TABLE data (
                      id  INTEGER PRIMARY KEY,
                      x   REAL
                      )", &[]).unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (0, 1.2)", &[]).unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (1, 2.3)", &[]).unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (2, 3.4)", &[]).unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (3, 4.5)", &[]).unwrap();
        conn
    }

    fn single_int_column_no_missing() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute("CREATE TABLE data (
                      id  INTEGER PRIMARY KEY,
                      x   INTEGER
                      )", &[]).unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (0, 1)", &[]).unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (1, 2)", &[]).unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (2, 3)", &[]).unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (3, 4)", &[]).unwrap();
        conn
    }

    #[test]
    fn read_continuous_data_with_no_missing() {
        let conn = single_real_column_no_missing();

        let data = sel_continuous_data(&"x", &"data", &conn);

        assert_eq!(data.len(), 4);
        assert_relative_eq!(data[0], 1.2, epsilon=10E-10);
        assert_relative_eq!(data[1], 2.3, epsilon=10E-10);
        assert_relative_eq!(data[2], 3.4, epsilon=10E-10);
        assert_relative_eq!(data[3], 4.5, epsilon=10E-10);
    }

    #[test]
    fn continuous_data_should_be_ordered_by_id() {
        let conn = single_real_column_no_missing();

        // Add data out of order (by id)
        conn.execute("INSERT INTO data (id, x) VALUES (5, 5.5)", &[]).unwrap();
        conn.execute("INSERT INTO data (id, x) VALUES (4, 4.4)", &[]).unwrap();

        let data = sel_continuous_data(&"x", &"data", &conn);

        assert_eq!(data.len(), 6);
        assert_relative_eq!(data[0], 1.2, epsilon=10E-10);
        assert_relative_eq!(data[1], 2.3, epsilon=10E-10);
        assert_relative_eq!(data[2], 3.4, epsilon=10E-10);
        assert_relative_eq!(data[3], 4.5, epsilon=10E-10);
        assert_relative_eq!(data[4], 4.4, epsilon=10E-10);
        assert_relative_eq!(data[5], 5.5, epsilon=10E-10);
    }

    #[test]
    fn read_cateegorical_data_with_no_missing() {
        let conn = single_int_column_no_missing();

        let data = sel_categorical_data(&"x", &"data", &conn);

        assert_eq!(data.len(), 4);
        assert_eq!(data[0], 1);
        assert_eq!(data[1], 2);
        assert_eq!(data[2], 3);
        assert_eq!(data[3], 4);
    }
}
