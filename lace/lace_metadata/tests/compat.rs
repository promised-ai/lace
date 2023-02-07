use std::path::Path;

#[test]
fn v1_to_latest() {
    let v1_satellites = Path::new("resources").join("satellites-v1.lace");
    let md = lace_metadata::load_metadata(v1_satellites, None).unwrap();

    assert!(md.data.is_some());
    assert!(md.rng.is_some());
}
