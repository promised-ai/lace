
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Codebook {
    pub table_name: String,
    pub metadata: Vec<MetaData>,
}


impl Codebook {
    pub fn new(table_name: String, metadata: Vec<MetaData>) -> Self {
        Codebook { table_name: table_name, metadata: metadata }
    }

    pub fn ids_are_unique(&self) -> bool {
        let mut ids: Vec<usize> = Vec::new();
        for md in &self.metadata {
            match md {
                &MetaData::Column {ref id, .. } if ids.contains(&id) => return false,
                &MetaData::Column {ref id, .. } => ids.push(*id),
                _ => (),
            }
        }

        if ids.is_empty() {
            panic!("No column metadata in codebook for {}", self.table_name)
        } else {
            true
        }
    }

    pub fn zip_col_metadata(&self) -> Vec<(usize, String, ColMetadata)> {
        let mut output = Vec::new();
        for md in &self.metadata {
            match md {
                &MetaData::Column{ref id, ref name, ref colmd} => {
                    output.push((*id, name.clone(), colmd.clone()));
                },
                _ => (),
            }
        }
        output
    }

    pub fn state_alpha(&self) -> Option<f64> {
        let alpha_opt = self.metadata.iter().find(|md| {
            match md {
                MetaData::StateAlpha { alpha } => true,
                _ => false,
            }
        });
        match alpha_opt {
            Some(MetaData::StateAlpha { alpha }) => Some(*alpha),
            Some(_) => panic!("Found wrong type"),
            None => None,
        }
    }
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ColMetadata {
    Continuous {
        m: f64,
        r: f64,
        s: f64,
        v: f64,
    },
    Categorical {
        alpha: f64,
        k: usize,
    },
    Binary {
        a: f64,
        b: f64,
    }
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MetaData{
    Column {
        id: usize,
        name: String,
        colmd: ColMetadata,
    },
    StateAlpha {
        alpha: f64
    },
    ViewAlpha {
        alpha: f64
    },
}

impl MetaData {
    pub fn is_column(&self) -> bool {
        match &self {
            MetaData::Column {..} => true,
            _ => false,
        }
    }

    pub fn is_state_alpha(&self) -> bool {
        match &self {
            MetaData::StateAlpha {..} => true,
            _ => false,
        }
    }

    pub fn is_view_alpha(&self) -> bool {
        match &self {
            MetaData::ViewAlpha {..} => true,
            _ => false,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique_id_check_with_all_unique_should_pass() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column { id: 0,
                                     name: "0".to_string(),
                                     colmd: colmd.clone() };
        let md1 = MetaData::Column { id: 1,
                                     name: "1".to_string(),
                                     colmd: colmd.clone() };
        let md2 = MetaData::Column { id: 2,
                                     name: "2".to_string(),
                                     colmd: colmd.clone() };
        let md3 = MetaData::StateAlpha { alpha: 1.0 };
        let md4 = MetaData::ViewAlpha { alpha: 1.0 };

        let metadata = vec![md0, md1, md2, md3, md4];

        let codebook = Codebook::new("table".to_string(), metadata);

        assert!(codebook.ids_are_unique());
    }

    #[test]
    fn unique_id_check_with_duplicates_should_fail() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column { id: 0,
                                     name: "0".to_string(),
                                     colmd: colmd.clone() };
        let md1 = MetaData::Column { id: 2,
                                     name: "1".to_string(),
                                     colmd: colmd.clone() };
        let md2 = MetaData::Column { id: 2,
                                     name: "2".to_string(),
                                     colmd: colmd.clone() };
        let md3 = MetaData::StateAlpha { alpha: 1.0 };
        let md4 = MetaData::ViewAlpha { alpha: 1.0 };

        let metadata = vec![md0, md1, md2, md3, md4];

        let codebook = Codebook::new("table".to_string(), metadata);

        assert!(!codebook.ids_are_unique());
    }

    #[test]
    fn unique_id_check_with_one_column_should_pass() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column { id: 0,
                                     name: "0".to_string(),
                                     colmd: colmd.clone() };
        let md1 = MetaData::StateAlpha { alpha: 1.0 };
        let md2 = MetaData::ViewAlpha { alpha: 1.0 };

        let metadata = vec![md0, md1, md2];

        let codebook = Codebook::new("table".to_string(), metadata);

        assert!(codebook.ids_are_unique());
    }

    #[test]
    #[should_panic]
    fn unique_id_check_with_no_columns_should_panic() {
        let md0 = MetaData::StateAlpha { alpha: 1.0 };
        let md1 = MetaData::ViewAlpha { alpha: 1.0 };

        let metadata = vec![md0, md1];

        let codebook = Codebook::new("table".to_string(), metadata);
        let _u = codebook.ids_are_unique();
    }

    #[test]
    fn zip_col_metadata_should_return_an_etry_for_each_column() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column { id: 0,
                                     name: "0".to_string(),
                                     colmd: colmd.clone() };
        let md1 = MetaData::Column { id: 2,
                                     name: "1".to_string(),
                                     colmd: colmd.clone() };
        let md2 = MetaData::StateAlpha { alpha: 1.0 };
        let md3 = MetaData::ViewAlpha { alpha: 1.0 };

        let metadata = vec![md0, md1, md2, md3];

        let codebook = Codebook::new("table".to_string(), metadata);
        let colmds = codebook.zip_col_metadata();

        assert_eq!(colmds.len(), 2);
        assert_eq!(colmds[0].0, 0);
        assert_eq!(colmds[1].0, 2);
    }

    #[test]
    fn get_state_alpha() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column { id: 0,
                                     name: "0".to_string(),
                                     colmd: colmd.clone() };
        let md1 = MetaData::Column { id: 2,
                                     name: "1".to_string(),
                                     colmd: colmd.clone() };
        let md2 = MetaData::StateAlpha { alpha: 2.3 };
        let md3 = MetaData::ViewAlpha { alpha: 1.0 };

        let metadata = vec![md0, md1, md2, md3];

        let codebook = Codebook::new("table".to_string(), metadata);

        assert!(codebook.state_alpha().is_some());
        assert_relative_eq!(codebook.state_alpha().unwrap(), 2.3, epsilon=10E-10);
    }
}
