pub mod sqlite;
pub mod traits;

pub enum Source {
    Sqlite,
    Postgres,
    Csv
}
