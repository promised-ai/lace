use misc::minmax;


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Codebook {
    pub table_name: String,
    pub metadata: Vec<MetaData>,
}


impl Codebook {
    pub fn new(table_name: String, metadata: Vec<MetaData>) -> Self {
        Codebook { table_name: table_name, metadata: metadata }
    }

    // FIXME: change to validate IDs
    pub fn validate_ids(&self) -> Result<(), &str> {
        let mut ids: Vec<usize> = Vec::new();
        for md in &self.metadata {
            match md {
                &MetaData::Column {ref id, .. } if ids.contains(&id) => {
                    return Err("IDs not unique")
                },
                &MetaData::Column {ref id, .. } => ids.push(*id),
                _ => (),
            }
        }

        if ids.is_empty() {
            Err("No column metadata")
        } else {
            let (min_id, max_id) = minmax(&ids);
            if min_id != 0 || max_id != ids.len() -1 {
                Err("IDs must span 0, 1, ..., n_cols-1")
            } else {
                Ok(())
            }
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
        output.sort_by_key(|(id, _, _)| *id);
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
    fn validate_id_with_properly_formed_ids_should_pass() {
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

        assert!(codebook.validate_ids().is_ok());
    }

    #[test]
    fn validate_ids_with_duplicates_should_fail() {
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

        assert!(codebook.validate_ids().is_err());
    }

    #[test]
    fn validate_ids_with_one_column_should_pass_if_id_is_0() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column { id: 0,
                                     name: "0".to_string(),
                                     colmd: colmd.clone() };

        let metadata = vec![md0];
        let codebook = Codebook::new("table".to_string(), metadata);

        assert!(codebook.validate_ids().is_ok());
    }

    #[test]
    fn validate_ids_with_one_column_should_fail_if_id_is_not_0() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column { id: 1,
                                     name: "0".to_string(),
                                     colmd: colmd.clone() };

        let metadata = vec![md0];
        let codebook = Codebook::new("table".to_string(), metadata);

        assert!(codebook.validate_ids().is_err());
    }

    #[test]
    fn validate_ids_with_no_columns_should_pfail() {
        let md0 = MetaData::StateAlpha { alpha: 1.0 };
        let md1 = MetaData::ViewAlpha { alpha: 1.0 };

        let metadata = vec![md0, md1];

        let codebook = Codebook::new("table".to_string(), metadata);
        assert!(codebook.validate_ids().is_err());
    }

    #[test]
    fn validate_ids_with_bad_id_span_should_fail_1() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column { id: 1,
                                     name: "0".to_string(),
                                     colmd: colmd.clone() };
        let md1 = MetaData::Column { id: 2,
                                     name: "1".to_string(),
                                     colmd: colmd.clone() };
        let md2 = MetaData::Column { id: 3,
                                     name: "2".to_string(),
                                     colmd: colmd.clone() };

        let metadata = vec![md0, md1, md2];
        let codebook = Codebook::new("table".to_string(), metadata);

        assert!(codebook.validate_ids().is_err());
    }

    #[test]
    fn validate_ids_with_bad_id_span_should_fail_2() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column { id: 0,
                                     name: "0".to_string(),
                                     colmd: colmd.clone() };
        let md1 = MetaData::Column { id: 1,
                                     name: "1".to_string(),
                                     colmd: colmd.clone() };
        let md2 = MetaData::Column { id: 3,
                                     name: "2".to_string(),
                                     colmd: colmd.clone() };

        let metadata = vec![md0, md1, md2];
        let codebook = Codebook::new("table".to_string(), metadata);

        assert!(codebook.validate_ids().is_err());
    }

    #[test]
    fn zip_col_metadata_should_return_an_etry_for_each_column() {
        let colmd = ColMetadata::Binary { a: 1.0, b: 2.0 };
        let md0 = MetaData::Column { id: 0,
                                     name: "0".to_string(),
                                     colmd: colmd.clone() };
        let md1 = MetaData::Column { id: 1,
                                     name: "1".to_string(),
                                     colmd: colmd.clone() };
        let md2 = MetaData::StateAlpha { alpha: 1.0 };
        let md3 = MetaData::ViewAlpha { alpha: 1.0 };

        let metadata = vec![md1, md2, md0, md3];

        let codebook = Codebook::new("table".to_string(), metadata);
        let colmds = codebook.zip_col_metadata();

        assert_eq!(colmds.len(), 2);
        assert_eq!(colmds[0].0, 0);
        assert_eq!(colmds[1].0, 1);
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
