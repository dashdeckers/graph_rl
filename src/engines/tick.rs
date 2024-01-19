use {
    crate::{
        agents::{
            RunMode,
            Algorithm,
            OffPolicyAlgorithm,
        },
        envs::{
            Environment,
            TensorConvertible,
        },
    },
    anyhow::Result,
    tracing::warn,
    candle_core::{
        Device,
        Tensor,
    },
    rand::{
        thread_rng,
        Rng,
    },
};


/// Run a single tick / step of an environment with any algorithm.
///
/// # Arguments
///
/// * `env` - The environment to step in.
/// * `agent` - The agent to step with.
/// * `device` - The device to run on.
pub fn tick<Alg, Env, Obs, Act>(
    env: &mut Env,
    agent: &mut Alg,
    mode: RunMode,
    device: &Device,
) -> Result<()>
where
    Env: Environment<Action = Act, Observation = Obs>,
    Alg: Algorithm,
    Obs: Clone + TensorConvertible,
    Act: Clone + TensorConvertible,
{
    let state = &<Obs>::to_tensor(env.current_observation(), device)?;
    let action = agent.actions(state, mode)?;
    let step = env.step(<Act>::from_tensor_pp(action))?;

    if step.terminated || step.truncated {
        env.reset(thread_rng().gen::<u64>())?;
    }

    let x = (step.reward, step.terminated, step.truncated);
    warn!("Environment has ticked with {x:?}");

    Ok(())
}

/// Run a single tick / step of an environment with an off-policy algorithm.
///
/// This function also calls `remember` on the agent.
///
/// # Arguments
///
/// * `env` - The environment to step in.
/// * `agent` - The agent to step with.
/// * `device` - The device to run on.
pub fn tick_off_policy<Alg, Env, Obs, Act>(
    env: &mut Env,
    agent: &mut Alg,
    mode: RunMode,
    device: &Device,
) -> Result<()>
where
    Env: Environment<Action = Act, Observation = Obs>,
    Alg: Algorithm + OffPolicyAlgorithm,
    Obs: Clone + TensorConvertible,
    Act: Clone + TensorConvertible,
{
    let state = &<Obs>::to_tensor(env.current_observation(), device)?;
    let action = &agent.actions(state, mode)?;
    let step = env.step(<Act>::from_tensor_pp(action.clone()))?;

    agent.remember(
        state,
        action,
        &Tensor::new(vec![step.reward], device)?,
        &<Obs>::to_tensor(step.observation, device)?,
        &Tensor::new(vec![step.terminated as u8], device)?,
        &Tensor::new(vec![step.truncated as u8], device)?,
    );

    if step.terminated || step.truncated {
        env.reset(thread_rng().gen::<u64>())?;
    }

    let x = (step.reward, step.terminated, step.truncated);
    warn!("Environment has ticked (off policy) with {x:?}");

    Ok(())
}
