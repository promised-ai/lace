//! Utilities for the satellites example
use crate::examples::IndexConversionError;
use std::convert::TryInto;

/// Row names for the animals data set
#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Column {
    CountryOfOperator,
    Users,
    Purpose,
    ClassOfOrbit,
    TypeOfOrbit,
    PerigeeKm,
    ApogeeKm,
    Eccentricity,
    PeriodMinutes,
    LaunchMassKg,
    DryMassKg,
    PowerWatts,
    DateOfLaunch,
    ExpectedLifetime,
    CountryOfContractor,
    LaunchSite,
    LaunchVehicle,
    SourceUsedForOrbitalData,
    LongitudeRadiansOfGeo,
    InclinationRadians,
}

impl Column {
    pub fn ix(self) -> usize {
        self.into()
    }
}

impl From<Column> for usize {
    fn from(col: Column) -> Self {
        col as Self
    }
}

impl TryInto<Column> for usize {
    type Error = IndexConversionError;
    fn try_into(self) -> Result<Column, Self::Error> {
        match self {
            0 => Ok(Column::CountryOfOperator),
            1 => Ok(Column::Users),
            2 => Ok(Column::Purpose),
            3 => Ok(Column::ClassOfOrbit),
            4 => Ok(Column::TypeOfOrbit),
            5 => Ok(Column::PerigeeKm),
            6 => Ok(Column::ApogeeKm),
            7 => Ok(Column::Eccentricity),
            8 => Ok(Column::PeriodMinutes),
            9 => Ok(Column::LaunchMassKg),
            10 => Ok(Column::DryMassKg),
            11 => Ok(Column::PowerWatts),
            12 => Ok(Column::DateOfLaunch),
            13 => Ok(Column::ExpectedLifetime),
            14 => Ok(Column::CountryOfContractor),
            15 => Ok(Column::LaunchSite),
            16 => Ok(Column::LaunchVehicle),
            17 => Ok(Column::SourceUsedForOrbitalData),
            18 => Ok(Column::LongitudeRadiansOfGeo),
            19 => Ok(Column::InclinationRadians),
            _ => Err(IndexConversionError::ColumnIndexOutOfBounds {
                col_ix: self,
                n_cols: 20,
            }),
        }
    }
}
