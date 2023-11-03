//! Wrappers around the Python API of Gymnasium (the new version of OpenAI gym)
use pyo3::prelude::*;
use pyo3::types::PyDict;
use anyhow::{Result, Error, anyhow};

use super::{Step, VectorConvertible};


fn w(res: PyErr) -> Error {
    anyhow!(res)
}

pub fn gym_create_env(
    name: &str,
) -> Result<(PyObject, Vec<usize>, Vec<usize>)> {
    Python::with_gil(|py| {
        let gym = py.import("gymnasium")?;
        let make = gym.getattr("make")?;
        let env = make.call1((name,))?;
        let action_space = env.getattr("action_space")?;
        let action_space = action_space.getattr("shape")?.extract()?;
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

pub fn gym_reset_env<O>(
    env: &PyObject,
    seed: u64,
) -> Result<O>
where
    O: VectorConvertible
{
    Python::with_gil(|py| {
        let kwargs = PyDict::new(py);
        kwargs.set_item("seed", seed)?;
        let observation = env.call_method(py, "reset", (), Some(kwargs))?;
        Ok(O::from_vec(observation.as_ref(py).get_item(0)?.extract::<Vec<f64>>()?))
    })
    .map_err(w)
}

pub fn gym_step_env<O, A>(
    env: &PyObject,
    action: A,
    goal_aware: bool,
) -> Result<Step<O, A>>
where
    O: VectorConvertible,
    A: pyo3::IntoPy<pyo3::Py<pyo3::PyAny>> + Clone
{
    let (observation, reward, terminated, truncated) = Python::with_gil(|py| {
        let step = env.call_method(py, "step", (vec![action.clone()],), None)?;
        let step = step.as_ref(py);
        let observation = if goal_aware {
            panic!("Not implemented yet")
        } else {
            O::from_vec(step.get_item(0)?.extract::<Vec<f64>>()?)
        };
        let reward: f64 = step.get_item(1)?.extract()?;
        let terminated: bool = step.get_item(2)?.extract()?;
        let truncated: bool = step.get_item(3)?.extract()?;
        Ok((observation, reward, terminated, truncated))
    })
    .map_err(w)?;
    Ok(Step {
        observation,
        action,
        reward,
        terminated,
        truncated,
    })
}

