use lace_data::Category;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::Hash;
use thiserror::Error;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(
    into = "BTreeMap<usize, T>",
    try_from = "BTreeMap<usize, T>",
    rename_all = "snake_case"
)]
pub struct CategoryMap<T>
where
    T: Hash + Clone + Eq + Default + Ord,
{
    to_cat: Vec<T>,
    to_ix: HashMap<T, usize>,
}

impl<T> CategoryMap<T>
where
    T: Hash + Clone + Eq + Default + Ord,
{
    pub fn len(&self) -> usize {
        self.to_cat.len()
    }

    pub fn is_empty(&self) -> bool {
        self.to_cat.is_empty()
    }

    pub fn ix(&self, cat: &T) -> Option<usize> {
        self.to_ix.get(cat).cloned()
    }

    pub fn category(&self, ix: usize) -> T {
        self.to_cat[ix].clone()
    }

    pub fn contains_cat(&self, cat: &T) -> bool {
        self.to_ix.contains_key(cat)
    }

    pub(crate) fn add(&mut self, value: T) {
        self.to_ix.insert(value.clone(), self.to_cat.len());
        self.to_cat.push(value);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValueMap {
    String(CategoryMap<String>),
    UInt(usize),
    Bool,
}

pub struct CategoryIter<'t> {
    map: &'t ValueMap,
    ix: usize,
}

impl<'t> CategoryIter<'t> {
    pub fn new(map: &'t ValueMap) -> Self {
        Self { map, ix: 0 }
    }
}

impl<'t> Iterator for CategoryIter<'t> {
    type Item = Category;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ix == self.map.len() {
            None
        } else {
            let value = self.map.category(self.ix);
            self.ix += 1;
            Some(value)
        }
    }
}

impl ValueMap {
    pub fn new<T>(cats: BTreeSet<T>) -> Self
    where
        Self: From<BTreeSet<T>>,
    {
        cats.into()
    }

    pub fn len(&self) -> usize {
        match self {
            Self::String(inner) => inner.len(),
            Self::UInt(k) => *k,
            Self::Bool => 2,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::String(inner) => inner.is_empty(),
            Self::UInt(k) => *k == 0,
            Self::Bool => false,
        }
    }

    /// Get the usize index of the category if it exists
    ///
    /// # Examples
    ///
    /// ```
    /// # use lace_data::Category;
    /// # use std::collections::BTreeSet;
    /// # use lace_codebook::ValueMap;
    /// let mut cats: BTreeSet<String> = BTreeSet::new();
    ///
    /// cats.insert("B".into());
    /// cats.insert("C".into());
    /// cats.insert("A".into());
    ///
    /// let value_map = ValueMap::new(cats);
    ///
    /// assert_eq!(value_map.ix(&Category::String("A".into())), Some(0));
    /// assert_eq!(value_map.ix(&Category::String("B".into())), Some(1));
    /// assert_eq!(value_map.ix(&Category::String("C".into())), Some(2));
    /// assert_eq!(value_map.ix(&Category::String("D".into())), None);
    /// ```
    pub fn ix(&self, cat: &Category) -> Option<usize> {
        match (self, cat) {
            (Self::String(map), Category::String(ref x)) => map.ix(x),
            (Self::UInt(k), Category::UInt(ref x)) => {
                if (*x as usize) < *k {
                    Some(*x as usize)
                } else {
                    None
                }
            }
            (Self::Bool, Category::Bool(x)) => Some(*x as usize),
            _ => None,
        }
    }

    /// Get the category associated with an index
    ///
    /// # Examples
    ///
    /// ```
    /// # use lace_data::Category;
    /// # use std::collections::BTreeSet;
    /// # use lace_codebook::ValueMap;
    /// let mut cats: BTreeSet<String> = BTreeSet::new();
    ///
    /// cats.insert("B".into());
    /// cats.insert("C".into());
    /// cats.insert("A".into());
    ///
    /// let value_map = ValueMap::new(cats);
    ///
    /// assert_eq!(value_map.category(0), Category::String("A".into()));
    /// assert_eq!(value_map.category(1), Category::String("B".into()));
    /// assert_eq!(value_map.category(2), Category::String("C".into()));
    /// ```
    pub fn category(&self, ix: usize) -> Category {
        match self {
            Self::String(inner) => Category::String(inner.category(ix)),
            Self::UInt(k) => {
                if ix < *k {
                    Category::UInt(ix as u32)
                } else {
                    panic!(
                        "index {ix} is too large for U32 map with length {k}"
                    );
                }
            }
            Self::Bool => match ix {
                0 => Category::Bool(false),
                1 => Category::Bool(true),
                _ => panic!("{ix} is too large for boolean map"),
            },
        }
    }

    pub fn contains_cat(&self, cat: &Category) -> bool {
        match (self, cat) {
            (Self::String(map), Category::String(x)) => map.contains_cat(x),
            (Self::UInt(k), Category::UInt(x)) => (*x as usize) < *k,
            (Self::Bool, Category::Bool(_)) => true,
            _ => false,
        }
    }

    pub fn iter(&self) -> CategoryIter {
        CategoryIter::new(self)
    }

    /// Determine whether a value map is an extended version of this value map
    ///
    /// # Examples
    ///
    /// ```
    /// # use lace_codebook::ValueMap;
    /// use std::collections::BTreeSet;
    /// use lace_data::Category;
    ///
    /// let mut cats: BTreeSet<String> = BTreeSet::new();
    ///
    /// cats.insert("B".into());
    /// cats.insert("C".into());
    /// cats.insert("A".into());
    ///
    /// let value_map_1 = ValueMap::new(cats.clone());
    ///
    /// assert!(value_map_1.is_extended(&value_map_1));
    ///
    /// cats.insert("D".into());
    ///
    /// let value_map_2 = ValueMap::new(cats);
    ///
    /// assert!(value_map_1.len() < value_map_2.len());
    /// assert!(value_map_1.is_extended(&value_map_2));
    /// assert!(!value_map_2.is_extended(&value_map_1));
    /// ```
    ///
    /// Integer valuemap
    ///
    /// ```
    /// # use lace_codebook::ValueMap;
    /// let value_map_1 = ValueMap::UInt(2);
    /// let value_map_2 = ValueMap::UInt(3);
    /// let value_map_3 = ValueMap::UInt(4);
    ///
    /// assert!(value_map_1.is_extended(&value_map_1));
    /// assert!(value_map_1.is_extended(&value_map_2));
    /// assert!(value_map_1.is_extended(&value_map_3));
    ///
    /// assert!(!value_map_2.is_extended(&value_map_1));
    /// assert!(value_map_2.is_extended(&value_map_2));
    /// assert!(value_map_2.is_extended(&value_map_3));
    ///
    /// assert!(!value_map_3.is_extended(&value_map_1));
    /// assert!(!value_map_3.is_extended(&value_map_2));
    /// assert!(value_map_3.is_extended(&value_map_3));
    /// ```
    ///
    /// ```
    /// # use lace_codebook::ValueMap;
    /// let value_map = ValueMap::Bool;
    ///
    /// assert!(value_map.is_extended(&value_map));
    /// ```
    pub fn is_extended(&self, other: &Self) -> bool {
        // TODO: DRY arms!
        match (self, other) {
            (Self::String(a), Self::String(b)) => {
                if b.len() < a.len() {
                    return false;
                }
                a.to_cat
                    .iter()
                    .zip(b.to_cat.iter())
                    .all(|(ai, bi)| ai == bi)
            }
            (Self::UInt(k_a), Self::UInt(k_b)) => k_b >= k_a,
            (Self::Bool, Self::Bool) => true,
            _ => false,
        }
    }

    /// Extend this `ValueMap`
    pub fn extend(
        &mut self,
        extension: ValueMapExtension,
    ) -> Result<(), ValueMapExtensionError> {
        match (self, extension) {
            (ValueMap::String(map), ValueMapExtension::String(additions)) => {
                additions.into_iter().for_each(|x| map.add(x));
                Ok(())
            }
            (
                ValueMap::UInt(ref mut cur_max),
                ValueMapExtension::UInt { new_max },
            ) => {
                if new_max > *cur_max {
                    *cur_max = new_max;
                }
                Ok(())
            }
            (vm, cat) => {
                let value_map_type = match vm {
                    ValueMap::String(_) => "string",
                    ValueMap::UInt(_) => "u32",
                    ValueMap::Bool => "bool",
                }
                .to_string();

                let extension_type = match cat {
                    ValueMapExtension::String(_) => "string",
                    ValueMapExtension::UInt { .. } => "u32",
                }
                .to_string();
                Err(ValueMapExtensionError::ExtensionOfDifferingType(
                    value_map_type,
                    extension_type,
                ))
            }
        }
    }
}

/// Errors from a failed `ValueMap.extend`
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Error)]
#[serde(rename_all = "snake_case")]
pub enum ValueMapExtensionError {
    #[error("ValueMap of type '{0}' cannot be extended with a type '{1}'")]
    ExtensionOfDifferingType(String, String),
}

impl TryFrom<Vec<String>> for ValueMap {
    type Error = String;

    fn try_from(cats: Vec<String>) -> Result<Self, Self::Error> {
        let to_ix: HashMap<String, usize> = cats
            .iter()
            .cloned()
            .enumerate()
            .map(|(ix, cat)| (cat, ix))
            .collect();

        if to_ix.len() == cats.len() {
            let cat_map = CategoryMap {
                to_ix,
                to_cat: cats,
            };
            Ok(Self::String(cat_map))
        } else {
            Err(String::from("Duplicate entries"))
        }
    }
}

impl From<BTreeSet<String>> for ValueMap {
    fn from(mut cats: BTreeSet<String>) -> Self {
        let k = cats.len();

        let mut to_cat = Vec::with_capacity(k);
        let mut to_ix = HashMap::with_capacity(k);

        let mut ix: usize = 0;
        while let Some(cat) = cats.pop_first() {
            to_cat.push(cat.clone());
            to_ix.insert(cat, ix);
            ix += 1;
        }

        let inner = CategoryMap { to_cat, to_ix };
        ValueMap::String(inner)
    }
}

impl<T> From<BTreeSet<T>> for CategoryMap<T>
where
    T: Hash + Clone + Eq + Ord + Default,
{
    fn from(mut set: BTreeSet<T>) -> Self {
        let k = set.len();

        let mut to_cat = Vec::with_capacity(k);
        let mut to_ix = HashMap::with_capacity(k);

        let mut ix: usize = 0;
        while let Some(cat) = set.pop_first() {
            to_cat.push(cat.clone());
            to_ix.insert(cat, ix);
            ix += 1;
        }

        Self { to_cat, to_ix }
    }
}

impl<T> From<CategoryMap<T>> for BTreeMap<usize, T>
where
    T: Hash + Clone + Eq + Default + Ord,
{
    fn from(mut value_map: CategoryMap<T>) -> Self {
        value_map.to_cat.drain(..).enumerate().collect()
    }
}

impl<T> TryFrom<BTreeMap<usize, T>> for CategoryMap<T>
where
    T: Hash + Clone + Eq + Default + Ord,
{
    type Error = String;

    fn try_from(mut map: BTreeMap<usize, T>) -> Result<Self, Self::Error> {
        let k = map.len();

        // fill to_cat with a dummy value so we can insert via indexing
        let mut to_cat = vec![T::default(); k];
        let mut to_ix = HashMap::new();

        while let Some((ix, cat)) = map.pop_first() {
            if ix < k {
                to_cat[ix] = cat.clone();
                if to_ix.insert(cat, ix).is_some() {
                    return Err(format!("Category {ix} is a duplicate"));
                }
            } else {
                return Err(format!("Category index {ix} exceeds the number of categories ({k})"));
            }
        }

        Ok(Self { to_ix, to_cat })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValueMapExtension {
    String(HashSet<String>),
    UInt { new_max: usize },
}

impl ValueMapExtension {
    pub fn new_string() -> Self {
        Self::String(HashSet::new())
    }

    pub fn new_uint() -> Self {
        Self::UInt { new_max: 0 }
    }

    pub fn extend(
        &mut self,
        category: Category,
    ) -> Result<(), ValueMapExtensionError> {
        match (self, category) {
            (ValueMapExtension::String(set), Category::String(x)) => {
                set.insert(x);
                Ok(())
            }
            (ValueMapExtension::UInt { new_max }, Category::UInt(x)) => {
                let x = x as usize;
                if x >= *new_max {
                    *new_max = x + 1;
                }
                Ok(())
            }
            (vm, cat) => {
                let value_map_type = match vm {
                    ValueMapExtension::String(_) => "string",
                    ValueMapExtension::UInt { .. } => "u32",
                }
                .to_string();
                let extension_type = match cat {
                    Category::Bool(_) => "bool",
                    Category::UInt(_) => "u32",
                    Category::String(_) => "string",
                }
                .to_string();
                Err(ValueMapExtensionError::ExtensionOfDifferingType(
                    value_map_type,
                    extension_type,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_valuemap_u32_iterator() {
        let vm = ValueMap::UInt(2);
        let cats: Vec<Category> = vm.iter().collect();
        assert_eq!(cats, &[Category::UInt(0), Category::UInt(1)]);
    }

    #[test]
    fn expected_valuemap_bool_iterator() {
        let vm = ValueMap::Bool;
        let cats: Vec<Category> = vm.iter().collect();
        assert_eq!(cats, &[Category::Bool(false), Category::Bool(true)]);
    }

    #[test]
    fn expected_valuemap_string_iterator() {
        let values = BTreeSet::from_iter([
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
        ]);
        let vm = ValueMap::new(values);
        let cats: Vec<Category> = vm.iter().collect();
        assert_eq!(
            cats,
            &[
                Category::String("A".to_string()),
                Category::String("B".to_string()),
                Category::String("C".to_string()),
            ]
        );
    }
}
