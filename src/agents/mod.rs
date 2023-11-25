mod ddpg;
mod ddpg_sgm;

pub mod configs;
pub use ddpg::DDPG;
pub use ddpg_sgm::DDPG_SGM;


use crate::{
    envs::Environment,
    components::ReplayBuffer,
    RunMode,
};
use {
    ordered_float::OrderedFloat,
    petgraph::{
        stable_graph::StableGraph,
        Undirected,
    },
    candle_core::{
        Tensor,
        Result,
        Device,
    }
};

pub trait Algorithm {
    type Config;

    fn from_config(
        device: &Device,
        config: &Self::Config,
        size_state: usize,
        size_action: usize,
    ) -> Result<Box<Self>>;

    fn run_mode(&self) -> RunMode;
    fn set_run_mode(&mut self, mode: RunMode);

    fn actions(&mut self, state: &Tensor) -> Result<Tensor>;
    fn train(&mut self) -> Result<()>;
}

pub trait OffPolicyAlgorithm: Algorithm {
    fn remember(
        &mut self,
        state: &Tensor,
        action: &Tensor,
        reward: &Tensor,
        next_state: &Tensor,
        terminated: &Tensor,
        truncated: &Tensor,
    );

    fn replay_buffer(&self) -> &ReplayBuffer;
}

pub trait SgmAlgorithm<Env: Environment>: Algorithm {
    fn plan(&self) -> &Vec<Env::Observation>;
    fn graph(&self) -> &StableGraph<Env::Observation, OrderedFloat<f64>, Undirected>;
    fn construct_graph(&mut self);
}