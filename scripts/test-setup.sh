#!/bin/bash

set -e

BRAID_TEST_DIR=braid-tests

mkdir $BRAID_TEST_DIR

VALID_DIR_1=$BRAID_TEST_DIR/savedata.valid.1.braid

# Generate a test save directory
mkdir $VALID_DIR_1
touch $VALID_DIR_1/0.state
touch $VALID_DIR_1/1.state
touch $VALID_DIR_1/2.state
touch $VALID_DIR_1/test.codebook
touch $VALID_DIR_1/test.data

NO_CODEBOOK_DIR=$BRAID_TEST_DIR/savedata.no.codebook.braid

# Generate a test save directory
mkdir $NO_CODEBOOK_DIR
touch $NO_CODEBOOK_DIR/0.state
touch $NO_CODEBOOK_DIR/1.state
touch $NO_CODEBOOK_DIR/test.data

NO_DATA_DIR=$BRAID_TEST_DIR/savedata.no.data.braid

# Generate a test save directory
mkdir $NO_DATA_DIR
touch $NO_DATA_DIR/0.state
touch $NO_DATA_DIR/1.state
touch $NO_DATA_DIR/2.state
touch $NO_DATA_DIR/3.state
touch $NO_DATA_DIR/test.codebook
