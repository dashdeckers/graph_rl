use std::fmt::Debug;
use candle_core::Tensor;
use anyhow::Result;


mod gym_wrappers;
pub mod point_env;
pub mod pendulum;



#[derive(Debug)]
pub struct Step<S, A> {
    pub state: S,
    pub action: A,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
}
pub trait Environment {
    type Config;
    type Action: Debug + Clone + From<Tensor> + Into<Tensor> + From<Vec<f64>>;
    type State: Debug + Clone + From<Tensor> + Into<Tensor>;

    fn new(config: Self::Config) -> Result<Box<Self>>;
    fn reset(&mut self, seed: u64) -> Result<Self::State>;
    fn step(&mut self, action: Self::Action) -> Result<Step<Self::State, Self::Action>>;
    fn action_space(&self) -> usize;
    fn observation_space(&self) -> &[usize];
}

