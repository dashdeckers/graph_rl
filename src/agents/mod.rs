mod ddpg;
mod ddpg_sgm;

pub use ddpg::DDPG;
pub use ddpg_sgm::DDPG_SGM;


use {
    crate::{
        envs::Environment,
        components::ReplayBuffer,
    },
    ordered_float::OrderedFloat,
    petgraph::{
        stable_graph::StableGraph,
        Undirected,
    },
    candle_core::{
        Tensor,
        Result,
        Device,
    },
    std::fmt::Display,
};


/// The execution mode of an agent is either training or testing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    Train,
    Test,
}

impl Display for RunMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunMode::Train => write!(f, "Train"),
            RunMode::Test => write!(f, "Test"),
        }
    }
}

pub trait Algorithm {
    type Config;

    fn config(&self) -> &Self::Config;
    fn from_config(
        device: &Device,
        config: &Self::Config,
        size_state: usize,
        size_action: usize,
    ) -> Result<Box<Self>>;

    fn actions(
        &mut self,
        state: &Tensor,
        mode: RunMode,
    ) -> Result<Tensor>;

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
    fn set_from_config(&mut self, config: &Self::Config);
}