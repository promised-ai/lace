use serde::{Deserialize, Serialize};

#[derive(
    Clone, Hash, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case", untagged)]
pub enum Category {
    Bool(bool),
    U8(u8),
    String(String),
}

impl Category {
    pub fn new<T: Into<Category>>(cat: T) -> Self {
        cat.into()
    }

    /// Convert the category to a u8 or panic if it cannot be converted.
    pub fn as_u8_or_panic(self) -> u8 {
        match self {
            Category::Bool(x) => x as u8,
            Category::U8(x) => x,
            Category::String(x) => {
                panic!("Cannot convert Category '{x}' to u8")
            }
        }
    }
}

impl From<bool> for Category {
    fn from(value: bool) -> Self {
        Category::Bool(value)
    }
}

impl From<u8> for Category {
    fn from(value: u8) -> Self {
        Category::U8(value)
    }
}

impl From<String> for Category {
    fn from(value: String) -> Self {
        Category::String(value)
    }
}

impl From<&str> for Category {
    fn from(value: &str) -> Self {
        Category::String(String::from(value))
    }
}
