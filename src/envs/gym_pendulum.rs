use candle_core::{Tensor, Device};
use pyo3::prelude::*;
use anyhow::Result;
use ordered_float::OrderedFloat;

use super::{Environment, Step, TensorConvertible, VectorConvertible, DistanceMeasure};
use super::gym_wrappers::{gym_create_env, gym_reset_env, gym_step_env};


pub struct PendulumEnv {
    env: PyObject,
    current_observation: Option<PendulumState>,
    action_space: Vec<usize>,
    observation_space: Vec<usize>,
}


pub struct PendulumConfig {
    name: String,
}
impl Default for PendulumConfig {
    fn default() -> Self {
        Self { name: "Pendulum-v1".to_owned() }
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PendulumAction {
    // Torque applied to the free end of the pendulum
    tau: OrderedFloat<f64>,
}
impl VectorConvertible for PendulumAction {
    fn from_vec(value: Vec<f64>) -> Self {
        // Make sure the number of elements in the Vec makes sense
        debug_assert!(value.len() == 1);
        Self {
            // Preprocess the action
            tau: OrderedFloat((2.0 * value[0]).clamp(-2.0, 2.0)),
        }
    }
    fn to_vec(value: Self) -> Vec<f64> {
        vec![*value.tau]
    }
}
impl TensorConvertible for PendulumAction {
    fn from_tensor(value: Tensor) -> Self {
        let values = value
            .squeeze(0).unwrap()
            .to_vec1::<f64>().unwrap();
        Self::from_vec(values)
    }
    fn to_tensor(value: Self, device: &Device) -> candle_core::Result<Tensor> {
        Tensor::new(Self::to_vec(value), device)
    }
}
// Convert PendulumAction into PyAny
impl IntoPy<PyObject> for PendulumAction {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.tau.into_py(py)
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PendulumState {
    // The (x, y) coordinates of the free end of the pendulum
    x: OrderedFloat<f64>,
    y: OrderedFloat<f64>,
    // The angular velocity of the pendulum
    theta: OrderedFloat<f64>,
}
impl VectorConvertible for PendulumState {
    fn from_vec(value: Vec<f64>) -> Self {
        // Make sure the number of elements in the Vec makes sense
        debug_assert!(value.len() == 3);
        Self {
            x: OrderedFloat(value[0]),
            y: OrderedFloat(value[1]),
            theta: OrderedFloat(value[2]),
        }
    }
    fn to_vec(value: Self) -> Vec<f64> {
        vec![*value.x, *value.y, *value.theta]
    }
}
impl TensorConvertible for PendulumState {
    fn from_tensor(value: Tensor) -> Self {
        let values = value
            .squeeze(0).unwrap()
            .to_vec1::<f64>().unwrap();
        Self::from_vec(values)
    }
    fn to_tensor(value: Self, device: &Device) -> candle_core::Result<Tensor> {
        Tensor::new(Self::to_vec(value), device)
    }
}
impl DistanceMeasure for PendulumState {
    fn distance(_s1: &Self, _s2: &Self) -> f64 {
        // 1.0
        todo!()
    }
}



impl Environment for PendulumEnv {
    type Config = PendulumConfig;
    type Action = PendulumAction;
    type Observation = PendulumState;

    fn new(config: Self::Config) -> Result<Box<Self>> {
        let (env, action_space, observation_space) = gym_create_env(&config.name)?;
        Ok(Box::new(Self {
            env,
            current_observation: None,
            action_space,
            observation_space,
        }))
    }

    fn reset(&mut self, seed: u64) -> Result<Self::Observation> {
        let obs = gym_reset_env(&self.env, seed)?;
        self.current_observation = Some(obs);
        Ok(self.current_observation())
    }

    fn step(&mut self, action: Self::Action) -> Result<Step<Self::Observation, Self::Action>> {
        let step: Step<Self::Observation, Self::Action> = gym_step_env(&self.env, action, false)?;
        self.current_observation = Some(step.observation.clone());
        Ok(step)
    }

    fn action_space(&self) -> Vec<usize> {
        self.action_space.clone()
    }

    fn observation_space(&self) -> Vec<usize> {
        self.observation_space.clone()
    }

    fn current_observation(&self) -> Self::Observation {
        if let Some(obs) = self.current_observation.clone() {
            obs
        } else {
            panic!("Can't access current observation of Gym environments before calling reset or step")
        }
    }
}
