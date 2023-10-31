use candle_core::Tensor;
use pyo3::prelude::*;
use anyhow::Result;

use super::{Environment, Step};
use super::gym_wrappers::{gym_create_env, gym_reset_env, gym_step_env};


pub struct PendulumEnv {
    env: PyObject,
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
// Convert Vec<f64> into PendulumState
impl From<Vec<f64>> for PendulumAction {
    fn from(value: Vec<f64>) -> Self {
        Self {
            tau: value[0],
        }
    }
}
// // Convert Tensor into PendulumAction
// impl TryFrom<Tensor> for PendulumAction {
//     type Error = Error;
//     fn try_from(value: Tensor) -> std::result::Result<Self, Self::Error> {
//         let values = value
//             .squeeze(0)
//             .map_err(|e| anyhow!(e.to_string()))?
//             .to_vec1::<f64>()
//             .map_err(|e| anyhow!(e.to_string()))?;
//         Ok(Self::from(values))
//     }
// }
// // Convert PendulumState into Tensor
// impl TryFrom<PendulumAction> for Tensor {
//     type Error = Error;
//     fn try_from(value: PendulumAction) -> std::result::Result<Self, Self::Error> {
//         Self::new(&[value.tau], &candle_core::Device::Cpu)
//             .map_err(|e| anyhow!(e.to_string()))
//     }
// }
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
// // Convert Vec<f64> into PendulumState
// impl From<Vec<f32>> for PendulumState {
//     fn from(value: Vec<f32>) -> Self {
//         Self {
//             x: value[0] as f64,
//             y: value[1] as f64,
//             theta: value[2] as f64,
//         }
//     }
// }
// Convert Vec<f64> into PendulumState
impl From<Vec<f64>> for PendulumState {
    fn from(value: Vec<f64>) -> Self {
        Self {
            x: value[0],
            y: value[1],
            theta: value[2],
        }
    }
}
// // Convert Tensor into PendulumState
// impl TryFrom<Tensor> for PendulumState {
//     type Error = Error;
//     fn try_from(value: Tensor) -> std::result::Result<Self, Self::Error> {
//         let values = value
//             .squeeze(0)
//             .map_err(|e| anyhow!(e.to_string()))?
//             .to_vec1::<f64>()
//             .map_err(|e| anyhow!(e.to_string()))?;
//         Ok(Self::from(values))
//     }
// }
// // Convert PendulumState into Tensor
// impl TryFrom<PendulumState> for Tensor {
//     type Error = Error;
//     fn try_from(value: PendulumState) -> std::result::Result<Self, Self::Error> {
//         Self::new(&[value.x, value.y, value.theta], &candle_core::Device::Cpu)
//             .map_err(|e| anyhow!(e.to_string()))
//     }
// }
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
            action_space,
            observation_space,
        }))
    }

    fn reset(&mut self, seed: u64) -> Result<Self::State> {
        gym_reset_env(&self.env, seed)
    }

    fn step(&mut self, action: Self::Action) -> Result<Step<Self::State, Self::Action>> {
        gym_step_env(&self.env, action)
    }

    fn action_space(&self) -> usize {
        self.action_space
    }

    fn observation_space(&self) -> &[usize] {
        &self.observation_space
    }
}
