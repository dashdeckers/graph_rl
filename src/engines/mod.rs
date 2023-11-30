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
mod train;
mod tick;
mod gui_offpolicy;
mod gui_sgm;

pub use experiment::run_experiment_off_policy;
pub use train::training_loop_off_policy;
pub use tick::{tick, tick_off_policy};

pub use gui_offpolicy::OffPolicyGUI;
pub use gui_sgm::SgmGUI;
