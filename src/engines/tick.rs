use {
    crate::{
        agents::{
            Algorithm,
            configs::AlgorithmConfig,
        },
        envs::{
            Environment,
            TensorConvertible,
        },
    },
    anyhow::Result,
    tracing::warn,
    candle_core::Device,
    rand::{
        thread_rng,
        Rng,
    },
};


pub fn tick<Alg, Env, Obs, Act>(
    env: &mut Env,
    agent: &mut Alg,
    device: &Device,
) -> Result<()>
where
    Env: Environment<Action = Act, Observation = Obs>,
    Alg: Algorithm,
    Alg::Config: AlgorithmConfig,
    Obs: Clone + TensorConvertible,
    Act: Clone + TensorConvertible,
{
    let state = &<Obs>::to_tensor(env.current_observation(), device)?;
    let action = agent.actions(state)?;
    let step = env.step(<Act>::from_tensor_pp(action))?;

    if step.terminated || step.truncated {
        env.reset(thread_rng().gen::<u64>())?;
    }

    let x = (step.reward, step.terminated, step.truncated);
    warn!("Environment has ticked with {x:?}");

    Ok(())
}
