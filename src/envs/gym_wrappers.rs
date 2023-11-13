//! Wrappers around the Python API of Gymnasium (the new version of OpenAI gym)
use {
    std::collections::HashMap,
    super::{
        Step,
        VectorConvertible,
    },
    anyhow::{
        anyhow,
        Error,
        Result,
    },
    pyo3::{
        Python,
        PyErr,
        PyResult,
        PyAny,
        PyObject,
        types::{
            PyDict,
            PyTuple,
        },
        exceptions::PyKeyError,
    },
};

fn w(res: PyErr) -> Error {
    anyhow!(res)
}

fn pyerr(msg: &str, py: Python) -> PyErr {
    PyErr::from_value(PyKeyError::new_err(msg.to_owned()).value(py))
}

fn get_observation<O>(
    py: Python,
    obs: &PyAny,
    goal_aware: bool,
) -> PyResult<O>
where
    O: VectorConvertible,
{
    Ok(if goal_aware {
        let mut obs = obs.extract::<HashMap<String, Vec<f64>>>()?;
        let observation = obs.remove("observation").ok_or_else(|| pyerr("No observation!", py))?;
        let desired_goal = obs.remove("desired_goal").ok_or_else(|| pyerr("No desired_goal!", py))?;
        let achieved_goal = obs.remove("achieved_goal").ok_or_else(|| pyerr("No achieved_goal!", py))?;

        O::from_vec([observation, desired_goal, achieved_goal].concat())
    } else {
        O::from_vec(obs.extract::<Vec<f64>>()?)
    })
}

pub fn gym_create_env(
    name: &str,
    goal_aware: bool,
) -> Result<(PyObject, Vec<usize>, Vec<usize>)> {
    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        let version: String = sys.getattr("version")?.extract()?;
        let path: String = sys.getattr("executable")?.extract()?;
        println!("PYTHON VERSION: {version}");
        println!("PYTHON EXECUTABLE: {path}");
        let gym = py.import("gymnasium")?;
        let make = gym.getattr("make")?;
        let env = make.call1((name,))?;

        let action_space = env.getattr("action_space")?;
        let action_space = action_space.getattr("shape")?.extract()?;

        let observation_space = if goal_aware {
            let get_dict_space = |obs_space: &PyAny, name: &str| {
                let space = obs_space.call_method1("__getitem__", (name, ))?;
                space.getattr("shape")?.extract::<Vec<usize>>()
            };
            let obs_space = env.getattr("observation_space")?;
            let observation_space = get_dict_space(obs_space, "observation")?;
            let desired_goal_space = get_dict_space(obs_space, "desired_goal")?;
            let achieved_goal_space = get_dict_space(obs_space, "achieved_goal")?;

            vec![observation_space[0] + desired_goal_space[0] + achieved_goal_space[0]]
        } else {
            let observation_space = env.getattr("observation_space")?;
            observation_space.getattr("shape")?.extract()?
        };

        Ok((env.into(), action_space, observation_space))
    })
    .map_err(w)
}

pub fn gym_reset_env<O>(
    env: &PyObject,
    seed: u64,
    goal_aware: bool,
) -> Result<O>
where
    O: VectorConvertible,
{
    Python::with_gil(|py| {
        let kwargs = PyDict::new(py);
        kwargs.set_item("seed", seed)?;
        let observation = env.call_method(py, "reset", (), Some(kwargs))?;
        get_observation(py, observation.as_ref(py).get_item(0)?, goal_aware)
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
    A: VectorConvertible + Clone + std::fmt::Debug,
{
    let (observation, reward, terminated, truncated) = Python::with_gil(|py| {
        let action = PyTuple::new(py, <A>::to_vec(action.clone()).iter());
        let step = env.call_method(py, "step", (action, ), None)?;
        let step = step.as_ref(py);
        let observation = get_observation(py, step.get_item(0)?, goal_aware)?;
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
