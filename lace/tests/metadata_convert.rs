use std::path::PathBuf;

#[test]
fn read_v0() {
    let path = PathBuf::from("resources")
        .join("test")
        .join("metadata")
        .join("v0")
        .join("metadata.lace");
    let _metadata = lace::metadata::load_metadata(path);
}

#[test]
fn read_v1() {
    let path = PathBuf::from("resources")
        .join("test")
        .join("metadata")
        .join("v1")
        .join("metadata.lace");
    let _metadata = lace::metadata::load_metadata(path);
}
