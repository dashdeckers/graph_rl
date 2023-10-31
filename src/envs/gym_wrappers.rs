//! Wrappers around the Python API of Gymnasium (the new version of OpenAI gym)
use pyo3::prelude::*;
use pyo3::types::PyDict;
use anyhow::{Result, Error, anyhow};

use super::Step;


fn w(res: PyErr) -> Error {
    anyhow!(res)
    // Error::wrap(res)
}

pub fn gym_create_env(name: &str) -> Result<(PyObject, usize, Vec<usize>)> {
    Python::with_gil(|py| {
        let gym = py.import("gymnasium")?;
        let make = gym.getattr("make")?;
        let env = make.call1((name,))?;
        let action_space = env.getattr("action_space")?;
        let action_space = if let Ok(val) = action_space.getattr("n") {
            val.extract()?
        } else {
            let action_space: Vec<usize> = action_space.getattr("shape")?.extract()?;
            action_space[0]
        };
        let observation_space = env.getattr("observation_space")?;
        let observation_space = observation_space.getattr("shape")?.extract()?;
        Ok((
            env.into(),
            action_space,
            observation_space,
        ))
    })
    .map_err(w)
}

pub fn gym_reset_env<S: From<Vec<f64>>>(env: &PyObject, seed: u64) -> Result<S> {
    Ok(Python::with_gil(|py| {
        let kwargs = PyDict::new(py);
        kwargs.set_item("seed", seed)?;
        let state = env.call_method(py, "reset", (), Some(kwargs))?;
        state.as_ref(py).get_item(0)?.extract::<Vec<f64>>()
    })
    .map_err(w)?
    .into())
}

pub fn gym_step_env<S: From<Vec<f64>>, A: pyo3::IntoPy<pyo3::Py<pyo3::PyAny>> + Clone>(env: &PyObject, action: A) -> Result<Step<S, A>> {
    let (state, reward, terminated, truncated) = Python::with_gil(|py| {
        let step = env.call_method(py, "step", (vec![action.clone()],), None)?;
        let step = step.as_ref(py);
        let state: Vec<f64> = step.get_item(0)?.extract()?;
        let reward: f64 = step.get_item(1)?.extract()?;
        let terminated: bool = step.get_item(2)?.extract()?;
        let truncated: bool = step.get_item(3)?.extract()?;
        Ok((state, reward, terminated, truncated))
    })
    .map_err(w)?;
    Ok(Step {
        state: state.into(),
        action,
        reward,
        terminated,
        truncated,
    })
}



// /// An OpenAI Gym session.
// pub struct GymEnv {
//     env: PyObject,
//     action_space: usize,
//     observation_space: Vec<usize>,
// }
// impl GymEnv {
//     /// Creates a new session of the specified OpenAI Gym environment.
//     pub fn new(name: &str) -> Result<GymEnv> {
//         Python::with_gil(|py| {
//             let gym = py.import("gymnasium")?;
//             let make = gym.getattr("make")?;
//             let env = make.call1((name,))?;
//             let action_space = env.getattr("action_space")?;
//             let action_space = if let Ok(val) = action_space.getattr("n") {
//                 val.extract()?
//             } else {
//                 let action_space: Vec<usize> = action_space.getattr("shape")?.extract()?;
//                 action_space[0]
//             };
//             let observation_space = env.getattr("observation_space")?;
//             let observation_space = observation_space.getattr("shape")?.extract()?;
//             Ok(GymEnv {
//                 env: env.into(),
//                 action_space,
//                 observation_space,
//             })
//         })
//         .map_err(w)
//     }

//     /// Resets the environment, returning the observation tensor.
//     pub fn reset(&self, seed: u64) -> Result<Tensor> {
//         let state: Vec<f64> = Python::with_gil(|py| {
//             let kwargs = PyDict::new(py);
//             kwargs.set_item("seed", seed)?;
//             let state = self.env.call_method(py, "reset", (), Some(kwargs))?;
//             state.as_ref(py).get_item(0)?.extract()
//         })
//         .map_err(w)?;
//         Tensor::new(state, &Device::Cpu)
//     }

//     /// Applies an environment step using the specified action.
//     pub fn step<A: pyo3::IntoPy<pyo3::Py<pyo3::PyAny>> + Clone>(
//         &self,
//         action: A,
//     ) -> Result<Step<A>> {
//         let (state, reward, terminated, truncated) = Python::with_gil(|py| {
//             let step = self.env.call_method(py, "step", (action.clone(),), None)?;
//             let step = step.as_ref(py);
//             let state: Vec<f64> = step.get_item(0)?.extract()?;
//             let reward: f64 = step.get_item(1)?.extract()?;
//             let terminated: bool = step.get_item(2)?.extract()?;
//             let truncated: bool = step.get_item(3)?.extract()?;
//             Ok((state, reward, terminated, truncated))
//         })
//         .map_err(w)?;
//         let state = Tensor::new(state, &Device::Cpu)?;
//         Ok(Step {
//             state,
//             action,
//             reward,
//             terminated,
//             truncated,
//         })
//     }

//     /// Returns the number of allowed actions for this environment.
//     pub fn action_space(&self) -> usize {
//         self.action_space
//     }

//     /// Returns the shape of the observation tensors.
//     pub fn observation_space(&self) -> &[usize] {
//         &self.observation_space
//     }
// }
