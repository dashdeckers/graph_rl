use {
    crate::{
        agents::{
            Algorithm,
            OffPolicyAlgorithm,
        },
        envs::{
            Environment,
            Sampleable,
            TensorConvertible,
        },
        configs::TrainConfig,
    },
    super::RunMode,
    anyhow::Result,
    candle_core::{
        Device,
        Tensor,
    },
    rand::Rng,
    tracing::warn,
};


/// Train a single run on an environment with an off-policy algorithm.
///
/// # Arguments
///
/// * `env` - The environment to train on.
/// * `alg` - The agent to train with.
/// * `config` - The configuration for the algorithm.
/// * `device` - The device to run on.
pub fn loop_off_policy<Alg, Env, Obs, Act>(
    env: &mut Env,
    alg: &mut Alg,
    config: TrainConfig,
    device: &Device,
) -> Result<(Vec<f64>, Vec<bool>)>
where
    Env: Environment<Action = Act, Observation = Obs>,
    Alg: Algorithm + OffPolicyAlgorithm,
    Obs: Clone + TensorConvertible,
    Act: Clone + TensorConvertible + Sampleable,
{
    warn!("action space: {:?}", env.action_space());
    warn!("observation space: {:?}", env.observation_space());

    let mut steps_taken = 0;
    let mut mc_returns = Vec::new();
    let mut successes = Vec::new();
    let mut rng = rand::thread_rng();

    for episode in 0..config.max_episodes() {
        let mut total_reward = 0.0;
        env.reset(rng.gen::<u64>())?;

        loop {
            let state = &<Obs>::to_tensor(env.current_observation(), device)?;

            // select an action, or randomly sample one
            let action = &if steps_taken < config.initial_random_actions() {
                <Act>::to_tensor(<Act>::sample(&mut rng, &env.action_domain()), device)?
            } else {
                alg.actions(state, config.run_mode())?
            };

            let step = env.step(<Act>::from_tensor_pp(action.clone()))?;
            total_reward += step.reward;
            steps_taken += 1;

            alg.remember(
                state,
                action,
                &Tensor::new(vec![step.reward], device)?,
                &<Obs>::to_tensor(step.observation, device)?,
                &Tensor::new(vec![step.terminated as u8], device)?,
                &Tensor::new(vec![step.truncated as u8], device)?,
            );

            if step.terminated || step.truncated {
                successes.push(step.terminated);
                break;
            }
        }

        warn!("episode {episode} with total reward of {total_reward}");
        mc_returns.push(total_reward);

        if let RunMode::Train = config.run_mode() {
            for _ in 0..config.training_iterations() {
                alg.train()?;
            }
        }
    }
    Ok((mc_returns, successes))
}