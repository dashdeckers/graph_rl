mod ddpg;
mod ddpg_sgm;

pub mod configs;
pub use ddpg::DDPG;
pub use ddpg_sgm::DDPG_SGM;


use crate::{
    components::ReplayBuffer,
    RunMode,
};
use candle_core::{
    Tensor,
    Result,
    Device,
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