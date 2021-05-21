pub mod latest;

pub trait MetadataVersion {
    fn metadata_version() -> u32;
}

#[macro_export]
macro_rules! impl_metadata_version {
    ($type:ty, $version:expr) => {
        impl MetadataVersion for $type {
            fn metadata_version() -> u32 {
                $version
            }
        }
    };
}
