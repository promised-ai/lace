mod helpers;

use helpers::{get_resp, post_resp};

use braid_server::api::obj::FType;
use braid_server::api::v1;
use warp::hyper::StatusCode;

#[test_log::test(tokio::test)]
async fn nstates() {
    let body = get_resp("/api/v1/nstates", None).await;
    let res: v1::NStatesResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.nstates, 4);
}

#[test_log::test(tokio::test)]
async fn shape() {
    let body = get_resp("/api/v1/shape", None).await;
    let res: v1::ShapeResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.n_rows, 50);
    assert_eq!(res.n_cols, 85);
}

#[test_log::test(tokio::test)]
async fn ftypes() {
    let body = get_resp("/api/v1/ftypes", None).await;
    let res: v1::FTypesResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.ftypes.len(), 85);
    assert!(res.ftypes.iter().all(|ftype| *ftype == FType::Categorical));
}

#[test_log::test(tokio::test)]
async fn ftype() {
    let body = get_resp("/api/v1/ftype/0", None).await;
    let res: v1::FTypeResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.ftype, FType::Categorical);
}

#[test_log::test(tokio::test)]
async fn diagnostics() {
    let body = get_resp("/api/v1/diagnostics", None).await;
    let res: v1::StateDiagnosticsResponse =
        serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.diagnostics.len(), 4);
}

#[test_log::test(tokio::test)]
async fn codebook() {
    let body = get_resp("/api/v1/codebook", None).await;
    let _res: v1::CodebookResponse =
        serde_json::from_str(body.as_str()).unwrap();
}

#[test_log::test(tokio::test)]
async fn single_depprob() {
    let query = "{\"col_pairs\": [[0, 1]]}";
    let body = post_resp("/api/v1/depprob", Some(query), None).await;
    let res: v1::DepprobResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.depprob.len(), 1);
}

#[test_log::test(tokio::test)]
async fn multi_depprob() {
    let query = "{\"col_pairs\": [[0, 1], [1, 2], [3, 3]]}";
    let body = post_resp("/api/v1/depprob", Some(query), None).await;
    let res: v1::DepprobResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.depprob.len(), 3);
}

#[test_log::test(tokio::test)]
async fn too_long_depprob_returns_status_400() {
    // Create a depprob query that has an over-the-limit number of col_pairs
    let mut query = String::from("{\"col_pairs\": [[0, 1], [1, 2], ");
    (0..10_000).for_each(|_| query.push_str("[0, 0],"));
    query.push_str("[3, 3]]}");

    let body = post_resp(
        "/api/v1/depprob",
        Some(query.as_str()),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    assert!(body.contains("error"));
}

#[test_log::test(tokio::test)]
async fn single_rowsim() {
    let query = "{\"row_pairs\": [[0, 1]], \"wrt\": []}";
    let body = post_resp("/api/v1/rowsim", Some(query), None).await;
    let res: v1::RowsimResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.rowsim.len(), 1);
}

#[test_log::test(tokio::test)]
async fn multi_rowsim() {
    let query = "{\"row_pairs\": [[0, 1], [1, 2], [3, 3]], \"wrt\": []}";
    let body = post_resp("/api/v1/rowsim", Some(query), None).await;
    let res: v1::RowsimResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.rowsim.len(), 3);
}

#[test_log::test(tokio::test)]
async fn too_long_rowsim_returns_status_400() {
    // Create a rowsim query that has an over-the-limit number of row_pairs
    let mut query = String::from("{\"row_pairs\": [[0, 1], [1, 2], ");
    (0..10_000).for_each(|_| query.push_str("[0, 0],"));
    query.push_str("[3, 3]], \"wrt\": []}");

    let body = post_resp(
        "/api/v1/rowsim",
        Some(query.as_str()),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    assert!(body.contains("error"));
}

#[test_log::test(tokio::test)]
async fn single_rowsim_col_weighted() {
    let query =
        "{\"row_pairs\": [[0, 1]], \"wrt\": [], \"col_weighted\": true}";
    let body = post_resp("/api/v1/rowsim", Some(query), None).await;
    let res: v1::RowsimResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.rowsim.len(), 1);
}

#[test_log::test(tokio::test)]
async fn multi_rowsim_col_weighted() {
    let query = "{\"row_pairs\": [[0, 1], [1, 2], [3, 3]], \"wrt\": [], \"col_weighted\": true}";
    let body = post_resp("/api/v1/rowsim", Some(query), None).await;
    let res: v1::RowsimResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.rowsim.len(), 3);
}

#[test_log::test(tokio::test)]
async fn too_long_rowsim_returns_status_400_col_weighted() {
    // Create a rowsim query that has an over-the-limit number of row_pairs
    let mut query = String::from("{\"row_pairs\": [[0, 1], [1, 2], ");
    (0..10_000).for_each(|_| query.push_str("[0, 0],"));
    query.push_str("[3, 3]], \"wrt\": [], \"col_weighted\": true}");

    let body = post_resp(
        "/api/v1/rowsim",
        Some(query.as_str()),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    assert!(body.contains("error"));
}

#[test_log::test(tokio::test)]
async fn single_mi() {
    let query = "{\"col_pairs\": [[0, 1]], \"n\": 1000, \"mi_type\": \"iqr\"}";
    let body = post_resp("/api/v1/mi", Some(query), None).await;
    let res: v1::MiResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.mi.len(), 1);
}

#[test_log::test(tokio::test)]
async fn multi_mi() {
    let query = "{\"col_pairs\": [[0, 1], [1, 2], [3, 3]], \
                 \"n\": 1000, \
                 \"mi_type\": \"iqr\"\
                 }";
    let body = post_resp("/api/v1/mi", Some(query), None).await;
    let res: v1::MiResponse = serde_json::from_str(body.as_str()).unwrap();
    assert_eq!(res.mi.len(), 3);
}

#[test_log::test(tokio::test)]
async fn too_long_mi_n_returns_status_400() {
    // number of samples too high
    let query = String::from(
        "{\"col_pairs\": [[0, 1]], \
         \"n\": 1000000,\
         \"mi_type\": \"iqr\"\
         }",
    );

    let body = post_resp(
        "/api/v1/mi",
        Some(query.as_str()),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    assert!(body.contains("error"));
}

#[test_log::test(tokio::test)]
async fn too_long_mi_pairs_retuns_status_400() {
    // (number of samples) * (number of col_pairs) too high
    let mut query = String::from("{\"col_pairs\": [[0, 1], [1, 2], [3, 3], ");
    (0..10_000).for_each(|_| query.push_str("[0, 0], "));
    query.push_str(
        " [3, 3]], \
         \"n\": 1000, \
         \"mi_type\": \"iqr\"\
         }",
    );

    let body = post_resp(
        "/api/v1/mi",
        Some(query.as_str()),
        Some(StatusCode::BAD_REQUEST),
    )
    .await;
    assert!(body.contains("error"));
}
