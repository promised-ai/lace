mod helpers;

#[cfg(test)]
mod test {
    use super::*;
    use std::io::Read;

    use warp::hyper::{header::CONTENT_TYPE, StatusCode};

    fn body_to_lines(body: Vec<u8>) -> Vec<Vec<String>> {
        use flate2::read::GzDecoder;
        let mut buf = String::new();

        let mut decoder = GzDecoder::new(body.as_slice());
        decoder.read_to_string(&mut buf).unwrap();

        buf.split('\n')
            .map(|line| {
                line.split(',')
                    .map(|x| x.to_owned())
                    .collect::<Vec<String>>()
            })
            .collect()
    }

    #[test_log::test(tokio::test)]
    async fn csv_download() {
        // this test checks whether the metadata file is generated on demand if
        // it hasn't been already generated by a mutating function.
        let filter = helpers::fixture(false);

        let (parts, body) = warp::test::request()
            .path("/api/v1/csv")
            .method("GET")
            .header("Accept-encoding", "gzip")
            .reply(&filter)
            .await
            .into_parts();

        assert_eq!(parts.status, StatusCode::OK);

        let lines = body_to_lines(body.to_vec());

        assert_eq!(lines.len(), 51);
        for line in lines.iter() {
            assert_eq!(line.len(), 86);
        }

        assert_eq!(lines[2][50], "1");
    }

    #[test_log::test(tokio::test)]
    async fn csv_download_after_data_update() {
        let filter = helpers::fixture(false);

        let (parts, body) = {
            let query = r#"{
                "rows": [
                    {"row_ix": "grizzly+bear",
                     "values" : [{
                        "col_ix": "hibernate",
                        "value": {"categorical": 0}
                    }]}
                ],
                "write_mode": {
                    "insert": "deny_new_rows_and_columns",
                    "overwrite": "allow",
                    "allow_extend_support": false
                }
            }"#;
            warp::test::request()
                .path("/api/v1/insert")
                .method("POST")
                .body(query)
                .header(CONTENT_TYPE, "application/json")
                .reply(&filter)
                .await;

            warp::test::request()
                .path("/api/v1/csv")
                .method("GET")
                .header("Accept-encoding", "gzip")
                .reply(&filter)
                .await
                .into_parts()
        };

        assert_eq!(parts.status, StatusCode::OK);

        let lines = body_to_lines(body.to_vec());

        assert_eq!(lines.len(), 51);
        for line in lines.iter() {
            assert_eq!(line.len(), 86);
        }

        // was "1" before update
        assert_eq!(lines[2][50], "0");
    }

    #[test_log::test(tokio::test)]
    async fn csv_download_after_append_column() {
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

        let filter = helpers::fixture(false);
        let (parts, body) = {
            warp::test::request()
                .path("/api/v1/insert")
                .method("POST")
                .body(query)
                .header(CONTENT_TYPE, "application/json")
                .reply(&filter)
                .await;

            warp::test::request()
                .path("/api/v1/csv")
                .method("GET")
                .header("Accept-encoding", "gzip")
                .reply(&filter)
                .await
                .into_parts()
        };

        assert_eq!(parts.status, StatusCode::OK);

        let lines = body_to_lines(body.to_vec());

        assert_eq!(lines.len(), 51);
        for line in lines.iter() {
            assert_eq!(line.len(), 87); // used to be 86
        }

        assert_eq!(lines[50][86], String::from("0"));
    }

    #[test_log::test(tokio::test)]
    async fn csv_download_after_append_row() {
        let filter = helpers::fixture(false);

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

        let (parts, body) = {
            warp::test::request()
                .path("/api/v1/insert")
                .method("POST")
                .body(query)
                .header(CONTENT_TYPE, "application/json")
                .reply(&filter)
                .await;

            warp::test::request()
                .path("/api/v1/csv")
                .method("GET")
                .header("Accept-encoding", "gzip")
                .reply(&filter)
                .await
                .into_parts()
        };

        assert_eq!(parts.status, StatusCode::OK);

        let lines = body_to_lines(body.to_vec());

        assert_eq!(lines.len(), 52); // used to be 51
        for line in lines.iter() {
            assert_eq!(line.len(), 86);
        }

        assert_eq!(lines[51][37], String::from("0"));
    }
}
