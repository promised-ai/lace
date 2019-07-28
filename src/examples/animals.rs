//! Utilities for the animals example

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

impl Into<usize> for Row {
    fn into(self) -> usize {
        self as usize
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

impl Into<usize> for Column {
    fn into(self) -> usize {
        self as usize
    }
}
