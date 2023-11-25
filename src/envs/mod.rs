mod gym_pendulum;
mod gym_pointmaze;
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
    gym_pointmaze::{
        PointMazeConfig,
        PointMazeEnv,
    },
    pointenv::{
        config::PointEnvConfig,
        point_env::PointEnv,
        reward::PointReward,
    },
};

pub trait TensorConvertible: VectorConvertible {
    fn from_tensor_pp(value: Tensor) -> Self;
    fn from_tensor(value: Tensor) -> Self;
    fn to_tensor(
        value: Self,
        device: &Device,
    ) -> candle_core::Result<Tensor>;
}

pub trait VectorConvertible {
    fn from_vec_pp(value: Vec<f64>) -> Self;
    fn from_vec(value: Vec<f64>) -> Self;
    fn to_vec(value: Self) -> Vec<f64>;
}

pub trait GoalAwareObservation {
    type State;
    type View;

    fn observation(&self) -> &Self::View;
    fn desired_goal(&self) -> &Self::State;
    fn achieved_goal(&self) -> &Self::State;

    fn set_observation(&mut self, value: &Self::View);
    fn set_desired_goal(&mut self, value: &Self::State);
    fn set_achieved_goal(&mut self, value: &Self::State);
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
    fn timelimit(&self) -> usize;
    fn action_space(&self) -> Vec<usize>;
    fn action_domain(&self) -> Vec<RangeInclusive<f64>>;
    fn observation_space(&self) -> Vec<usize>;
    fn observation_domain(&self) -> Vec<RangeInclusive<f64>>;
    fn current_observation(&self) -> Self::Observation;
    fn value_range(&self) -> (f64, f64);
    fn config(&self) -> Self::Config;
}

pub trait Renderable {
    fn render(
        &mut self,
        plot_ui: &mut PlotUi,
    );
}
