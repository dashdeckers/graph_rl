//! # Components
//!
//! This module contains the components that can be used to build an agent.
//!
//! ## Noise
//!
//! The `Noise` components are typically used to add noise to the actions of an
//! agent. For example, the [`OuNoise`] struct implements the Ornstein-Uhlenbeck
//! process, which is typically used in the [`crate::agents::DDPG`] algorithm.
//!
//! ## Replay Buffer
//!
//! The [`ReplayBuffer`] struct implements a replay buffer, which is typically
//! used in off-policy algorithms such as [`crate::agents::DDPG`].
//!
//! ## SGM
//!
//! The [`Sgm`] struct implements Sparse Graphical Memory, which is used in the
//! [`crate::agents::DDPG_SGM`] algorithm to build a sparse graph on top of the
//! replay buffer.

mod ou_noise;
mod replay_buffer;

pub mod sgm;
pub use ou_noise::OuNoise;
pub use replay_buffer::ReplayBuffer;
