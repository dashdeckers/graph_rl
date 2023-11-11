mod gym_pendulum;
mod gym_wrappers;
mod pointenv;

use {
    anyhow::Result,
    candle_core::{
        Device,
        Tensor,
    },
    egui_plot::PlotUi,
    rand::RngCore,
    std::ops::RangeInclusive,
};

pub use crate::envs::{
    gym_pendulum::{
        PendulumConfig,
        PendulumEnv,
    },
    pointenv::{
        config::PointEnvConfig,
        point_env::PointEnv,
    },
};

pub trait TensorConvertible: VectorConvertible {
    fn from_tensor(value: Tensor) -> Self;
    fn to_tensor(
        value: Self,
        device: &Device,
    ) -> candle_core::Result<Tensor>;
}

pub trait VectorConvertible {
    fn from_vec(value: Vec<f64>) -> Self;
    fn to_vec(value: Self) -> Vec<f64>;
}

pub trait Sampleable {
    fn sample(
        rng: &mut dyn RngCore,
        domain: &[RangeInclusive<f64>],
    ) -> Self;
}

pub trait DistanceMeasure {
    fn distance(
        s1: &Self,
        s2: &Self,
    ) -> f64;
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
    fn reset(
        &mut self,
        seed: u64,
    ) -> Result<Self::Observation>;
    fn step(
        &mut self,
        action: Self::Action,
    ) -> Result<Step<Self::Observation, Self::Action>>;
    fn action_space(&self) -> Vec<usize>;
    fn action_domain(&self) -> Vec<RangeInclusive<f64>>;
    fn observation_space(&self) -> Vec<usize>;
    fn observation_domain(&self) -> Vec<RangeInclusive<f64>>;
    fn current_observation(&self) -> Self::Observation;
    fn value_range(&self) -> (f64, f64);
}

pub trait Renderable {
    fn render(
        &mut self,
        plot_ui: &mut PlotUi,
    );
}
