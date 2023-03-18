use lace_data::Category;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(
    into = "HashMap<usize, Category>",
    try_from = "HashMap<usize, Category>"
)]
pub struct ValueMap {
    to_cat: Vec<Category>,
    to_ix: HashMap<Category, usize>,
}

impl ValueMap {
    /// Create a new value map
    pub fn new(set: BTreeSet<Category>) -> Self {
        Self::from(set)
    }

    pub fn len(&self) -> usize {
        self.to_cat.len()
    }

    pub fn is_empty(&self) -> bool {
        self.to_cat.is_empty()
    }

    /// Get the usize index of the category if it exists
    ///
    /// # Examples
    ///
    /// ```
    /// # use lace_data::Category;
    /// # use std::collections::BTreeSet;
    /// # use lace_codebook::ValueMap;
    /// let mut cats: BTreeSet<Category> = BTreeSet::new();
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
        self.to_ix.get(cat).cloned()
    }

    /// Get the category associated with an index
    ///
    /// # Examples
    ///
    /// ```
    /// # use lace_data::Category;
    /// # use std::collections::BTreeSet;
    /// # use lace_codebook::ValueMap;
    /// let mut cats: BTreeSet<Category> = BTreeSet::new();
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
        self.to_cat[ix].clone()
    }
}

impl From<BTreeSet<Category>> for ValueMap {
    fn from(mut set: BTreeSet<Category>) -> Self {
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

impl From<ValueMap> for HashMap<usize, Category> {
    fn from(mut value_map: ValueMap) -> Self {
        value_map.to_cat.drain(..).enumerate().collect()
    }
}

impl TryFrom<HashMap<usize, Category>> for ValueMap {
    type Error = String;

    fn try_from(
        mut map: HashMap<usize, Category>,
    ) -> Result<Self, Self::Error> {
        let k = map.len();

        // fill to_cat with a dummy value so we can insert via indexing
        let mut to_cat = vec![Category::Bool(false); k];
        let mut to_ix = HashMap::with_capacity(k);

        map.drain().try_for_each(|(ix, cat)| {
            if ix < k {
                to_cat[ix] = cat.clone();
                if to_ix.insert(cat, ix).is_some() {
                    Err(format!("Category {ix} is a duplicate"))
                } else {
                    Ok(())
                }
            } else {
                Err(format!("Category index {ix} exceeds the number of categories ({k})"))
            }
        })?;

        Ok(Self { to_ix, to_cat })
    }
}
