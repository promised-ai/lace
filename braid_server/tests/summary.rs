mod helpers;

use braid::NameOrIndex;
use braid_server::api::v1;
use helpers::post_resp;
use std::collections::BTreeMap;

#[test_log::test(tokio::test)]
async fn summary_single_index() {
    let query = r#"{ "col_ixs": [0] }"#;

    let body = post_resp("/api/v1/summary", Some(query), None).await;
    let summaries = {
        let mut res: v1::SummarizeColumnsResponse =
            serde_json::from_str(body.as_str()).unwrap();

        res.summaries.drain(..).collect::<BTreeMap<_, _>>()
    };

    assert_eq!(summaries.len(), 1);
    assert!(summaries.contains_key(&NameOrIndex::Index(0)));
}

#[test_log::test(tokio::test)]
async fn summary_single_name() {
    let query = r#"{ "col_ixs": ["swims"] }"#;

    let body = post_resp("/api/v1/summary", Some(query), None).await;
    let summaries = {
        let mut res: v1::SummarizeColumnsResponse =
            serde_json::from_str(body.as_str()).unwrap();

        res.summaries.drain(..).collect::<BTreeMap<_, _>>()
    };

    assert_eq!(summaries.len(), 1);
    assert!(summaries.contains_key(&NameOrIndex::Name("swims".into())));
}

#[test_log::test(tokio::test)]
async fn summary_multi() {
    let query = r#"{ "col_ixs": [0, 2] }"#;

    let body = post_resp("/api/v1/summary", Some(query), None).await;
    let summaries = {
        let mut res: v1::SummarizeColumnsResponse =
            serde_json::from_str(body.as_str()).unwrap();

        res.summaries.drain(..).collect::<BTreeMap<_, _>>()
    };

    assert_eq!(summaries.len(), 2);
    assert!(summaries.contains_key(&NameOrIndex::Index(0)));
    assert!(summaries.contains_key(&NameOrIndex::Index(2)));
}

#[test_log::test(tokio::test)]
async fn summary_multi_name() {
    let query = r#"{ "col_ixs": ["swims", "flys"] }"#;

    let body = post_resp("/api/v1/summary", Some(query), None).await;
    let summaries = {
        let mut res: v1::SummarizeColumnsResponse =
            serde_json::from_str(body.as_str()).unwrap();

        res.summaries.drain(..).collect::<BTreeMap<_, _>>()
    };

    assert_eq!(summaries.len(), 2);
    assert!(summaries.contains_key(&NameOrIndex::Name("swims".into())));
    assert!(summaries.contains_key(&NameOrIndex::Name("flys".into())));
}
