#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

bash $DIR/test-setup.sh
cargo test
bash $DIR/cleanup.sh

