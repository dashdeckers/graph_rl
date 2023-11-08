mod gym_wrappers;
mod pointenv;
mod gym_pendulum;

pub use crate::envs::{
    gym_pendulum::{
        PendulumEnv,
        PendulumConfig,
    },
    pointenv::{
        config::PointEnvConfig,
        point_env::PointEnv,
    },
};

use candle_core::{Tensor, Device};
use anyhow::Result;
use egui_plot::PlotUi;

pub trait TensorConvertible: VectorConvertible {
    fn from_tensor(value: Tensor) -> Self;
    fn to_tensor(value: Self, device: &Device) -> candle_core::Result<Tensor>;
}

pub trait VectorConvertible {
    fn from_vec(value: Vec<f64>) -> Self;
    fn to_vec(value: Self) -> Vec<f64>;
}

pub trait DistanceMeasure {
    fn distance(s1: &Self, s2: &Self) -> f64;
}


#[derive(Debug)]
pub struct Step<O, A> {
    pub observation: O,
    pub action: A,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
}
pub trait Environment {
    type Config;
    type Action;
    type Observation;

    fn new(config: Self::Config) -> Result<Box<Self>>;
    fn reset(&mut self, seed: u64) -> Result<Self::Observation>;
    fn step(&mut self, action: Self::Action) -> Result<Step<Self::Observation, Self::Action>>;
    fn action_space(&self) -> Vec<usize>;
    fn observation_space(&self) -> Vec<usize>;
    fn current_observation(&self) -> Self::Observation;
    fn episodic_reward_range(&self) -> (f64, f64);
}

pub trait Renderable {
    fn render(
        &mut self,
        plot_ui: &mut PlotUi,
    );
}
