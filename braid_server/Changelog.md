# Changelog

## 0.40.0
- Use `braid` v0.40.0
- Allow users to pass optional state indices to predict
- insert_data query now requires String column and row indices
- offload all validation to braid. Error messages will change, but should cover
    all edge cases

## 0.39.1
- Restore `--version` to `braid_server` command

## 0.39.0
- version numbers changed to align with braid 
- download route hidden behind `download` feature

## 0.25.1
- Always save new version of metadata when requesting download
- Temporary metadata files are created only for downloads and cleaned up
    afterward

## 0.25.0
- Metadata saves as a tarball of the expected metadata (metadata.braid.tar.gz)
    instead of a binary file that the user may not be able to use.

## 0.24.0
- server instance is immutable by default meaning no `update` or `insert_data`
    call are allowed
- bump braid to 0.36

## 0.23.1
- Add ability to lock the braid_server to a specific machine via a hardware ID

## 0.22.1
- Add `assignments` route to get column and row assignments for each state in
    an `Engine`
- Add ability to specify port (`--port`), JSON size limit (`--json-limit`), and
    file size limit (`--file-limit`) in CLI.

## 0.22.0
- Bump to braid 0.34.0 to fix a bug that causes a panic when a view weight in a
    conditional likelihood computation is zero

## 0.21.1
- Bump to braid 0.33.2 to fix a panic that occurs when predicting continuous
    values with many conditions (underflow)

## 0.21.0
- Update API to generate swagger docs

## 0.20.0
- Migrate to Rocket 0.5.0-rc.1
- Migrate to braid 0.33.0
- Updated insert_data api call to use `row_ix` and `col_ix` instead of
    `row_name` and `col_name`

## 0.19.0
- Implemented entire `braid::OracleT` and `braid::Engine` api

## 0.18.0
- upgrade to braid 0.32.0.

## 0.17.0
- upgrade to braid 0.31.0.
- All API calls explicitly return JSON content type

## 0.15.0
- upgrade to braid 0.27.0.
    + Changes insert_data api to require write_mode
    + NOTE: Extending support of categorical columns is not yet supported

## 0.14.0
- upgrade to braid 0.24.0. Breaks metadata.

## 0.13.0
- Upgrade to braid 23. Pre-baid-23 metadata no longer supported.
