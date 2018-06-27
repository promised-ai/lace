#!/bin/bash
set -x

REPORT_URL=http://bax.pythonanywhere.com
FILENAME=`date +%s`_quick.json
BRAID_REGRESSION_DIR=tmp

cargo build --release
echo ${APP_AUTH}
RUST_LOG=info target/release/braid regression -o $FILENAME resources/regression/configs/quick.yaml && \
curl -u ${APP_AUTH} -F "file=@$FILENAME;type=text/json" $REPORT_URL/upload
