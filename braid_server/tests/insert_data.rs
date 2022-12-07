// TODO: Write tests that allow a mutable state
mod helpers;
use std::convert::Infallible;

use braid::Datum;
use braid_server::api::v1::GetDataResponse;
use helpers::post_resp_client;

async fn get_datum<F, R>(
    client: &F,
    row_ix: usize,
    col_ix: usize,
) -> GetDataResponse
where
    F: warp::Filter<Extract = R, Error = Infallible> + 'static,
    R: warp::Reply,
{
    let query = format!(
        "{{
        \"ixs\": [
            [{}, {}]
        ]
    }}",
        row_ix, col_ix
    );

    let body = post_resp_client(
        client,
        "/api/v1/get_data",
        Some(query.as_str()),
        None,
    )
    .await;

    serde_json::from_str(body.as_str()).unwrap()
}

#[test_log::test(tokio::test)]
async fn insert_data() {
    let query = r#"{
        "rows": [
            {"row_ix": "dolphin",
             "values" : [{
                "col_ix": "swims",
                "value": {"categorical": 0}
            }]}
        ],
        "write_mode": {
            "insert": "deny_new_rows_and_columns",
            "overwrite": "allow",
            "allow_extend_support": false
        }
    }"#;

    let client = helpers::fixture(false);

    {
        let res = get_datum(&client, 49, 36).await;
        assert_eq!(res.values[0].2, Datum::Categorical(1));
    }

    let body =
        post_resp_client(&client, "/api/v1/insert", Some(query), None).await;
    let res = String::from(body.as_str());

    assert_eq!(res, String::from("{\"new_rows\":0,\"new_cols\":0}"));

    {
        let res = get_datum(&client, 49, 36).await;
        assert_eq!(res.values[0].2, Datum::Categorical(0));
    }
}

#[test_log::test(tokio::test)]
async fn insert_data_encrypted() {
    let query = r#"{
        "rows": [
            {"row_ix": "dolphin",
             "values" : [{
                "col_ix": "swims",
                "value": {"categorical": 0}
            }]}
        ],
        "write_mode": {
            "insert": "deny_new_rows_and_columns",
            "overwrite": "allow",
            "allow_extend_support": false
        }
    }"#;

    let client = helpers::fixture(true);

    {
        let res = get_datum(&client, 49, 36).await;
        assert_eq!(res.values[0].2, Datum::Categorical(1));
    }

    let body =
        post_resp_client(&client, "/api/v1/insert", Some(query), None).await;
    let res = String::from(body.as_str());

    assert_eq!(res, String::from("{\"new_rows\":0,\"new_cols\":0}"));

    {
        let res = get_datum(&client, 49, 36).await;
        assert_eq!(res.values[0].2, Datum::Categorical(0));
    }
}

#[test_log::test(tokio::test)]
async fn insert_new_column() {
    let query = r#"{
        "rows": [
            {"row_ix": "dolphin",
             "values" : [{
                "col_ix": "drinks+blood",
                "value": {"categorical": 0}
            }]}
        ],
        "write_mode": {
            "insert": "deny_new_rows",
            "overwrite": "deny",
            "allow_extend_support": false
        },
        "new_col_metadata": [{
            "name": "drinks+blood",
            "coltype": {
                "Categorical": {
                    "k": 2,
                    "hyper": {
                        "pr_alpha": {
                            "shape": 1.0,
                            "scale": 2.0
                        }
                    },
                    "prior": null,
                    "value_map": null
                }
            },
            "notes": null
        }]
    }"#;

    let client = helpers::fixture(false);
    let body =
        post_resp_client(&client, "/api/v1/insert", Some(query), None).await;
    let res = String::from(body.as_str());

    assert_eq!(res, String::from("{\"new_rows\":0,\"new_cols\":1}"));

    {
        // There are 50 rows and 85 columns in the original data, which means a
        // new row would have index 50 and a new column would have index 85.
        let res = get_datum(&client, 49, 85).await;
        assert_eq!(res.values[0].2, Datum::Categorical(0));
    }
}

#[test_log::test(tokio::test)]
async fn insert_new_row() {
    let query = r#"{
        "rows": [
            {"row_ix": "unicorn",
             "values" : [{
                "col_ix": "swims",
                "value": {"categorical": 0}
            }]}
        ],
        "write_mode": {
            "insert": "deny_new_columns",
            "overwrite": "deny",
            "allow_extend_support": false
        }
    }"#;

    let client = helpers::fixture(false);
    let body =
        post_resp_client(&client, "/api/v1/insert", Some(query), None).await;
    let res = String::from(body.as_str());

    assert_eq!(res, String::from("{\"new_rows\":1,\"new_cols\":0}"));

    {
        // There are 50 rows and 85 columns in the original data, which means a
        // new row would have index 50 and a new column would have index 85.
        let res = get_datum(&client, 50, 36).await;
        assert_eq!(res.values[0].2, Datum::Categorical(0));
    }
}

#[test_log::test(tokio::test)]
async fn insert_new_row_and_column() {
    let query = r#"{
        "rows": [{"row_ix": "unicorn",
            "values" : [{
                "col_ix": "drinks+blood",
                "value": {"categorical": 1}
            },{
                "col_ix": "swims",
                "value": {"categorical": 0}
            }]
        }],
        "write_mode": {
            "insert": "unrestricted",
            "overwrite": "deny",
            "allow_extend_support": false
        },
        "new_col_metadata": [{
            "name": "drinks+blood",
            "coltype": {
                "Categorical": {
                    "k": 2,
                    "hyper": {
                        "pr_alpha": {
                            "shape": 1.0,
                            "scale": 2.0
                        }
                    },
                    "prior": null,
                    "value_map": null
                }
            },
            "notes": null
        }]
    }"#;

    let client = helpers::fixture(false);
    let body =
        post_resp_client(&client, "/api/v1/insert", Some(query), None).await;
    let res = String::from(body.as_str());

    assert_eq!(res, String::from("{\"new_rows\":1,\"new_cols\":1}"));

    // There are 50 rows and 85 columns in the original data, which means a new
    // row would have index 50 and a new column would have index 85.
    {
        let res = get_datum(&client, 50, 36).await;
        assert_eq!(res.values[0].2, Datum::Categorical(0));
    }

    {
        let res = get_datum(&client, 50, 85).await;
        assert_eq!(res.values[0].2, Datum::Categorical(1));
    }
}
