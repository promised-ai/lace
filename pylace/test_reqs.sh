#!/bin/bash

set -eu

for py_ver in "3.10" "3.11" "3.12"; do
  rm -f uv.lock && uv run -p $py_ver --resolution lowest pytest -x > /dev/null
  rm -f uv.lock && uv run -p $py_ver pytest -x > /dev/null
done
echo "OK"