
/// Oracle answers questions
#[derive(Clone, Serialize, Deserialize)]
pub struct Oracle {
    /// Vector of states
    pub states: Vec<State>,
    /// Metadata for the rows and columns
    pub codebook: Codebook,
    pub data: DataStore,
}

impl Seer {
    /// Convert an `Engine` into an `Oracle`
    pub fn from_engine(engine: Engine) -> Self {
        let data = {
            let data_map = engine.states.values().nth(0).unwrap().clone_data();
            DataStore::new(data_map)
        };

        // TODO: would be nice to have a draining iterator on the states
        // rather than cloning them
        let states: Vec<State> = engine
            .states
            .values()
            .map(|state| {
                let mut state_clone = state.clone();
                state_clone.drop_data();
                state_clone
            })
            .collect();

        Oracle {
            data,
            states,
            codebook: engine.codebook,
        }
    }

    /// Load an Oracle from a .braid file
    pub fn load(dir: &Path) -> std::io::Result<Self> {
        let config = file_utils::load_file_config(dir).unwrap_or_default();
        let data = file_utils::load_data(dir, &config)?;
        let mut states = file_utils::load_states(dir, &config)?;
        let codebook = file_utils::load_codebook(dir)?;

        // Move states from map to vec
        let ids: Vec<usize> = states.keys().copied().collect();
        let states_vec =
            ids.iter().map(|id| states.remove(id).unwrap()).collect();

        Ok(Oracle {
            states: states_vec,
            codebook,
            data: DataStore::new(data),
        })
    }
}
