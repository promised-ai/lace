//! Utilities for the animals example
use std::convert::TryInto;

use crate::examples::IndexConversionError;
use crate::{ColumnIndex, NameOrIndex, RowIndex, TableIndex};

/// Row names for the animals data set
#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Row {
    Antelope,
    GrizzlyBear,
    KillerWhale,
    Beaver,
    Dalmatian,
    PersianCat,
    Horse,
    GermanShepherd,
    BlueWhale,
    SiameseCat,
    Skunk,
    Mole,
    Tiger,
    Hippopotamus,
    Leopard,
    Moose,
    SpiderMonkey,
    HumpbackWhale,
    Elephant,
    Gorilla,
    Ox,
    Fox,
    Sheep,
    Seal,
    Chimpanzee,
    Hamster,
    Squirrel,
    Rhinoceros,
    Rabbit,
    Bat,
    Giraffe,
    Wolf,
    Chihuahua,
    Rat,
    Weasel,
    Otter,
    Buffalo,
    Zebra,
    GiantPanda,
    Deer,
    Bobcat,
    Pig,
    Lion,
    Mouse,
    PolarBear,
    Collie,
    Walrus,
    Raccoon,
    Cow,
    Dolphin,
}

impl From<Row> for usize {
    fn from(row: Row) -> Self {
        row as usize
    }
}

impl From<Row> for RowIndex {
    fn from(row: Row) -> Self {
        RowIndex(NameOrIndex::Index(row.into()))
    }
}

impl From<Row> for TableIndex {
    fn from(row: Row) -> Self {
        TableIndex::Row(row.into())
    }
}

impl TryInto<Row> for usize {
    type Error = IndexConversionError;
    fn try_into(self) -> Result<Row, Self::Error> {
        match self {
            0 => Ok(Row::Antelope),
            1 => Ok(Row::GrizzlyBear),
            2 => Ok(Row::KillerWhale),
            3 => Ok(Row::Beaver),
            4 => Ok(Row::Dalmatian),
            5 => Ok(Row::PersianCat),
            6 => Ok(Row::Horse),
            7 => Ok(Row::GermanShepherd),
            8 => Ok(Row::BlueWhale),
            9 => Ok(Row::SiameseCat),
            10 => Ok(Row::Skunk),
            11 => Ok(Row::Mole),
            12 => Ok(Row::Tiger),
            13 => Ok(Row::Hippopotamus),
            14 => Ok(Row::Leopard),
            15 => Ok(Row::Moose),
            16 => Ok(Row::SpiderMonkey),
            17 => Ok(Row::HumpbackWhale),
            18 => Ok(Row::Elephant),
            19 => Ok(Row::Gorilla),
            20 => Ok(Row::Ox),
            21 => Ok(Row::Fox),
            22 => Ok(Row::Sheep),
            23 => Ok(Row::Seal),
            24 => Ok(Row::Chimpanzee),
            25 => Ok(Row::Hamster),
            26 => Ok(Row::Squirrel),
            27 => Ok(Row::Rhinoceros),
            28 => Ok(Row::Rabbit),
            29 => Ok(Row::Bat),
            30 => Ok(Row::Giraffe),
            31 => Ok(Row::Wolf),
            32 => Ok(Row::Chihuahua),
            33 => Ok(Row::Rat),
            34 => Ok(Row::Weasel),
            35 => Ok(Row::Otter),
            36 => Ok(Row::Buffalo),
            37 => Ok(Row::Zebra),
            38 => Ok(Row::GiantPanda),
            39 => Ok(Row::Deer),
            40 => Ok(Row::Bobcat),
            41 => Ok(Row::Pig),
            42 => Ok(Row::Lion),
            43 => Ok(Row::Mouse),
            44 => Ok(Row::PolarBear),
            45 => Ok(Row::Collie),
            46 => Ok(Row::Walrus),
            47 => Ok(Row::Raccoon),
            48 => Ok(Row::Cow),
            49 => Ok(Row::Dolphin),
            _ => Err(IndexConversionError::RowIndexOutOfBounds {
                row_ix: self,
                n_rows: 50,
            }),
        }
    }
}

/// Row names for the animals data set
#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Column {
    Black,
    White,
    Blue,
    Brown,
    Gray,
    Orange,
    Red,
    Yellow,
    Patches,
    Spots,
    Stripes,
    Furry,
    Hairless,
    Toughskin,
    Big,
    Small,
    Bulbous,
    Lean,
    Flippers,
    Hands,
    Hooves,
    Pads,
    Paws,
    Longleg,
    Longneck,
    Tail,
    Chewteeth,
    Meatteeth,
    Buckteeth,
    Strainteeth,
    Horns,
    Claws,
    Tusks,
    Smelly,
    Flys,
    Hops,
    Swims,
    Tunnels,
    Walks,
    Fast,
    Slow,
    Strong,
    Weak,
    Muscle,
    Bipedal,
    Quadrapedal,
    Active,
    Inactive,
    Nocturnal,
    Hibernate,
    Agility,
    Fish,
    Meat,
    Plankton,
    Vegetation,
    Insects,
    Forager,
    Grazer,
    Hunter,
    Scavenger,
    Skimmer,
    Stalker,
    Newworld,
    Oldworld,
    Arctic,
    Coastal,
    Desert,
    Bush,
    Plains,
    Forest,
    Fields,
    Jungle,
    Mountains,
    Ocean,
    Ground,
    Water,
    Tree,
    Cave,
    Fierce,
    Timid,
    Smart,
    Group,
    Solitary,
    Nestspot,
    Domestic,
}

impl From<Column> for usize {
    fn from(col: Column) -> Self {
        col as usize
    }
}

impl From<Column> for ColumnIndex {
    fn from(col: Column) -> Self {
        ColumnIndex(NameOrIndex::Index(col.into()))
    }
}

impl From<Column> for TableIndex {
    fn from(col: Column) -> Self {
        TableIndex::Column(col.into())
    }
}

impl TryInto<Column> for usize {
    type Error = IndexConversionError;
    fn try_into(self) -> Result<Column, Self::Error> {
        match self {
            0 => Ok(Column::Black),
            1 => Ok(Column::White),
            2 => Ok(Column::Blue),
            3 => Ok(Column::Brown),
            4 => Ok(Column::Gray),
            5 => Ok(Column::Orange),
            6 => Ok(Column::Red),
            7 => Ok(Column::Yellow),
            8 => Ok(Column::Patches),
            9 => Ok(Column::Spots),
            10 => Ok(Column::Stripes),
            11 => Ok(Column::Furry),
            12 => Ok(Column::Hairless),
            13 => Ok(Column::Toughskin),
            14 => Ok(Column::Big),
            15 => Ok(Column::Small),
            16 => Ok(Column::Bulbous),
            17 => Ok(Column::Lean),
            18 => Ok(Column::Flippers),
            19 => Ok(Column::Hands),
            20 => Ok(Column::Hooves),
            21 => Ok(Column::Pads),
            22 => Ok(Column::Paws),
            23 => Ok(Column::Longleg),
            24 => Ok(Column::Longneck),
            25 => Ok(Column::Tail),
            26 => Ok(Column::Chewteeth),
            27 => Ok(Column::Meatteeth),
            28 => Ok(Column::Buckteeth),
            29 => Ok(Column::Strainteeth),
            30 => Ok(Column::Horns),
            31 => Ok(Column::Claws),
            32 => Ok(Column::Tusks),
            33 => Ok(Column::Smelly),
            34 => Ok(Column::Flys),
            35 => Ok(Column::Hops),
            36 => Ok(Column::Swims),
            37 => Ok(Column::Tunnels),
            38 => Ok(Column::Walks),
            39 => Ok(Column::Fast),
            40 => Ok(Column::Slow),
            41 => Ok(Column::Strong),
            42 => Ok(Column::Weak),
            43 => Ok(Column::Muscle),
            44 => Ok(Column::Bipedal),
            45 => Ok(Column::Quadrapedal),
            46 => Ok(Column::Active),
            47 => Ok(Column::Inactive),
            48 => Ok(Column::Nocturnal),
            49 => Ok(Column::Hibernate),
            50 => Ok(Column::Agility),
            51 => Ok(Column::Fish),
            52 => Ok(Column::Meat),
            53 => Ok(Column::Plankton),
            54 => Ok(Column::Vegetation),
            55 => Ok(Column::Insects),
            56 => Ok(Column::Forager),
            57 => Ok(Column::Grazer),
            58 => Ok(Column::Hunter),
            59 => Ok(Column::Scavenger),
            60 => Ok(Column::Skimmer),
            61 => Ok(Column::Stalker),
            62 => Ok(Column::Newworld),
            63 => Ok(Column::Oldworld),
            64 => Ok(Column::Arctic),
            65 => Ok(Column::Coastal),
            66 => Ok(Column::Desert),
            67 => Ok(Column::Bush),
            68 => Ok(Column::Plains),
            69 => Ok(Column::Forest),
            70 => Ok(Column::Fields),
            71 => Ok(Column::Jungle),
            72 => Ok(Column::Mountains),
            73 => Ok(Column::Ocean),
            74 => Ok(Column::Ground),
            75 => Ok(Column::Water),
            76 => Ok(Column::Tree),
            77 => Ok(Column::Cave),
            78 => Ok(Column::Fierce),
            79 => Ok(Column::Timid),
            80 => Ok(Column::Smart),
            81 => Ok(Column::Group),
            82 => Ok(Column::Solitary),
            83 => Ok(Column::Nestspot),
            84 => Ok(Column::Domestic),
            _ => Err(IndexConversionError::ColumnIndexOutOfBounds {
                col_ix: self,
                n_cols: 85,
            }),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::examples::Example;
    use crate::OracleT;

    #[test]
    fn rows_convert_properly() {
        let oracle = Example::Animals.oracle().unwrap();

        for ix in 0..oracle.n_rows() {
            let row: Row = ix.try_into().unwrap();
            let row_ix: usize = row.into();
            assert_eq!(ix, row_ix);
        }
    }

    #[test]
    fn columns_convert_properly() {
        let oracle = Example::Animals.oracle().unwrap();

        for ix in 0..oracle.n_cols() {
            let col: Column = ix.try_into().unwrap();
            let col_ix: usize = col.into();
            assert_eq!(ix, col_ix);
        }
    }
}
