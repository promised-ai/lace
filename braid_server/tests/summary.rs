mod helpers;

use braid_server::api::v1;
use helpers::post_resp;

#[test_log::test(tokio::test)]
async fn summary_single() {
    let query = r#"{ "col_ixs": [0] }"#;

    let body = post_resp("/api/v1/summary", Some(query), None).await;
    let res: v1::SummarizeColumnsResponse =
        serde_json::from_str(body.as_str()).unwrap();

    assert_eq!(res.summaries.len(), 1);
    assert!(res.summaries.contains_key(&0));
}

#[test_log::test(tokio::test)]
async fn summary_multi() {
    let query = r#"{ "col_ixs": [0, 2] }"#;

    let body = post_resp("/api/v1/summary", Some(query), None).await;
    let res: v1::SummarizeColumnsResponse =
        serde_json::from_str(body.as_str()).unwrap();

    assert_eq!(res.summaries.len(), 2);
    assert!(res.summaries.contains_key(&0));
    assert!(res.summaries.contains_key(&2));
}
