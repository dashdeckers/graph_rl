//! # Engines
//!
//! This module contains the different engines that encapsulate the training
//! and testing logic of algorithms and environments.
//!
//! ## Experiments
//!
//! The experiment engines train an agent on an environment multiple times
//! with the same parameters and collect performance data for each run.
//!
//! ## Training
//!
//! The training loop engines are used for training an agent on an environment
//! once, for a number of episodes.
//!
//! ## Testing
//!
//! The testing, i.e. tick, engines are used for running a single timestep of
//! an algorithm on an environment, without any training.
//!
//! ## GUIs
//!
//! The GUI engines are used for visualizing the training and/or testing of
//! an algorithm on an environment using a graphical user interface.

mod experiment;
mod run;
mod tick;
mod gui_offpolicy;
mod gui_hgb;

pub use experiment::run_experiment_off_policy;
pub use run::loop_off_policy;
pub use tick::{tick, tick_off_policy};

pub use gui_offpolicy::OffPolicyGUI;
pub use gui_hgb::HgbGUI;

use {
    serde::{
        Serialize,
        Deserialize,
    },
    crate::{
        agents::Algorithm,
        envs::Environment,
    }
};

pub enum ParamAlg<Alg>
where
    Alg: Algorithm,
    Alg::Config: Clone + Serialize,
{
    AsAlgorithm(Alg),
    AsConfig(Alg::Config),
}

pub enum ParamEnv<Env, Obs, Act>
where
    Env: Environment<Action = Act, Observation = Obs>,
    Env::Config: Clone + Serialize,
{
    AsEnvironment(Env),
    AsConfig(Env::Config),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunMode {
    Train,
    Test,
}


use {
    anyhow::Result,
    tracing::Level,
    std::{
        fs::{File, create_dir_all},
        path::Path,
        sync::Arc,
    },
    tracing_subscriber::{
        fmt::{
            layer,
            writer::MakeWriterExt,
        },
        layer::SubscriberExt,
        util::SubscriberInitExt,
    },
};

pub fn setup_logging(
    path: &dyn AsRef<Path>,
    min_level: Option<Level>,
) -> Result<()> {
    let path = Path::new("data/").join(path);
    create_dir_all(path.as_path())?;
    let log_file = Arc::new(File::create(path.join("debug.log"))?);

    tracing_subscriber::registry()
        // File writer
        .with(
            layer()
                .with_writer(log_file.with_max_level(match min_level {
                    Some(level) => level,
                    None => Level::INFO,
                }))
                .with_ansi(false),
        )
        // Create and set Subscriber
        .init();

    Ok(())
}
