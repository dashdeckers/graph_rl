use {
    crate::{
        agents::{
            Algorithm,
            AlgorithmConfig,
        },
        envs::{
            Environment,
            Step,
            TensorConvertible,
        },
    },
    anyhow::Result,
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
    let do_step = |env: &mut Env, agent: &mut Alg, device: &Device| -> Result<Step<Obs, Act>> {
        let observation = &<Obs>::to_tensor(env.current_observation(), device)?;
        let action = agent.actions(observation)?;
        env.step(<Act>::from_tensor_pp(action))
    };

    if let Ok(step) = do_step(env, agent, device) {
        step
    } else {
        env.reset(thread_rng().gen::<u64>())?;
        do_step(env, agent, device)?
    };

    Ok(())
}
