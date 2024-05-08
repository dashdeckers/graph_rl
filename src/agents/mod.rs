mod ddpg;
mod ddpg_hgb;

pub use ddpg::DDPG;
pub use ddpg_hgb::DDPG_HGB;


use {
    crate::{
        engines::RunMode,
        envs::Environment,
        components::ReplayBuffer,
    },
    ordered_float::OrderedFloat,
    petgraph::{
        stable_graph::StableGraph,
        Directed,
    },
    candle_core::{
        Tensor,
        Result,
        Device,
    },
    std::path::Path,
};


pub trait SaveableAlgorithm: Algorithm {
    fn save<P: AsRef<Path> + ?Sized>(
        &self,
        path: &P,
        name: &str,
    ) -> Result<()>;

    fn load<P: AsRef<Path> + ?Sized>(
        &mut self,
        path: &P,
        name: &str,
    ) -> Result<()>;
}

pub trait Algorithm {
    type Config;

    fn config(&self) -> &Self::Config;
    fn override_config(
        &mut self,
        config: &Self::Config,
    );
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

pub trait HgbAlgorithm<Env: Environment>: Algorithm {
    fn plan(&self) -> &Vec<Env::Observation>;
    fn graph(&self) -> &StableGraph<Env::Observation, OrderedFloat<f64>, Directed>;
    fn clear_graph(&mut self);
    fn construct_graph(&mut self);
    // fn set_from_config(&mut self, config: &Self::Config);
}