# lace_metadata

Archive of the metadata (savefile) formats for lace. In charge of versioning
and conversion.

## Dev

The current version of the meta stays in `latest.rs`. If a metadata change
happens, everything in `latest.rs` goes to a version file, `v<x>.rs`. For
example if the first version of the metadata changes due to a change in the
`State` metadata, everything in `latest.rs` will go into `v1.rs`. `latest.rs`
will contain only metadata items that are different.

Implement `MetadataVersion` for everything. To make things easier, each file
should have a `const METADATA_VERSION: u32`.
