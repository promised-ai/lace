#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

crate_version() {
    toml_path=$1/Cargo.toml
    version=`grep -oEi "^version = \"(.+)\"" ${toml_path} | grep -Po "\d+\.\d+\.\d+"`
    echo $version
}

check_published() {
    crate_name=$1
    version=$2

    matches=`cloudsmith ls pkgs -k ${CLOUDSMITH_API_KEY} --query "name:${crate_name} version:${version}" redpoll/crates | grep -o "${crate_name}" -c`

    if [ "${matches}" -eq 0 ]; then
        echo "no"
    elif [ "${matches}" -eq 1 ]; then
        echo "yes"
    else
        echo "Package search should have returned 0 or 1, but returned ${matches}"
        exit 1
    fi
}

publish_crate() {
    crate_name=$1
    crate_path=$2
    version=`crate_version $crate_path`

    if [ `check_published $crate_name $version` = "no" ]; then
        echo "== Publishing ${crate_name} v${version} =="
        cd $crate_path
        cargo package -q
        cloudsmith push cargo \
            -k ${CLOUDSMITH_API_KEY} \
            redpoll/crates \
            $DIR/target/package/${crate_name}-${version}.crate
    else
        echo "-- ${crate_name} v${version} already published. Skipping. --"
    fi

}

publish_subcrate() {
    start_dir=`pwd`
    crate_name=$1
    crate_path=$DIR/$crate_name
    publish_crate $crate_name $crate_path
    cd $start_dir
}

publish_subcrate braid_consts
publish_subcrate braid_utils
publish_subcrate braid_flippers
publish_subcrate braid_stats
publish_subcrate braid_geweke
publish_subcrate braid_codebook

publish_crate braid $DIR
