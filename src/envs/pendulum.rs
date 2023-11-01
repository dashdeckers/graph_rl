use candle_core::Tensor;
use pyo3::prelude::*;
use anyhow::Result;

use super::{Environment, Step};
use super::gym_wrappers::{gym_create_env, gym_reset_env, gym_step_env};


pub struct PendulumEnv {
    env: PyObject,
    state: Option<PendulumState>,
    action_space: usize,
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


#[derive(Clone, Debug)]
pub struct PendulumAction {
    // Torque applied to the free end of the pendulum
    tau: f64,
}
// Convert Vec<f64> into PendulumAction
impl From<Vec<f64>> for PendulumAction {
    fn from(value: Vec<f64>) -> Self {
        assert!(value.len() == 1);
        Self {
            tau: value[0],
        }
    }
}
// Convert Vec<f64> into PendulumAction
impl From<PendulumAction> for Vec<f64> {
    fn from(value: PendulumAction) -> Self {
        vec![value.tau]
    }
}
// Convert Tensor into PendulumAction
impl From<Tensor> for PendulumAction {
    fn from(value: Tensor) -> Self {
        let values = value
            .squeeze(0).unwrap()
            .to_vec1::<f64>().unwrap();
        Self::from(values)
    }
}
// Convert PendulumAction into Tensor
impl From<PendulumAction> for Tensor {
    fn from(value: PendulumAction) -> Self {
        Self::new(&[value.tau], &candle_core::Device::Cpu).unwrap()
    }
}
// Convert PendulumAction into PyAny
impl IntoPy<PyObject> for PendulumAction {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.tau.into_py(py)
    }
}


#[derive(Clone, Debug)]
pub struct PendulumState {
    // The (x, y) coordinates of the free end of the pendulum
    x: f64,
    y: f64,
    // The angular velocity of the pendulum
    theta: f64,
}
// Convert Vec<f64> into PendulumState
impl From<Vec<f64>> for PendulumState {
    fn from(value: Vec<f64>) -> Self {
        assert!(value.len() == 3);
        Self {
            x: value[0],
            y: value[1],
            theta: value[2],
        }
    }
}
// Convert Vec<f64> into PendulumState
impl From<PendulumState> for Vec<f64> {
    fn from(value: PendulumState) -> Self {
        vec![value.x, value.y, value.theta]
    }
}
// Convert Tensor into PendulumState
impl From<Tensor> for PendulumState {
    fn from(value: Tensor) -> Self {
        let values = value
            .squeeze(0).unwrap()
            .to_vec1::<f64>().unwrap();
        Self::from(values)
    }
}
// Convert PendulumState into Tensor
impl From<PendulumState> for Tensor {
    fn from(value: PendulumState) -> Self {
        Self::new(&[value.x, value.y, value.theta], &candle_core::Device::Cpu).unwrap()
    }
}



impl Environment for PendulumEnv {
    type Config = PendulumConfig;
    type Action = PendulumAction;
    type State = PendulumState;

    fn new(config: Self::Config) -> Result<Box<Self>> {
        let (env, action_space, observation_space) = gym_create_env(&config.name)?;
        Ok(Box::new(Self {
            env,
            state: None,
            action_space,
            observation_space,
        }))
    }

    fn reset(&mut self, seed: u64) -> Result<Self::State> {
        let state: Self::State = gym_reset_env(&self.env, seed)?;
        self.state = Some(state.clone());
        Ok(state)
    }

    fn step(&mut self, action: Self::Action) -> Result<Step<Self::State, Self::Action>> {
        let step: Step<Self::State, Self::Action> = gym_step_env(&self.env, action)?;
        self.state = Some(step.state.clone());
        Ok(step)
    }

    fn action_space(&self) -> usize {
        self.action_space
    }

    fn observation_space(&self) -> &[usize] {
        &self.observation_space
    }

    fn current_state(&self) -> Self::State {
        self.state.clone().unwrap()
    }

    fn current_goal(&self) -> Self::State {
        panic!("Not implemented (yet?)")
    }
}
