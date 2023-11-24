use {
    crate::{
        agents::{
            Algorithm,
            OffPolicyAlgorithm,
            AlgorithmConfig,
        },
        envs::{
            Environment,
            Sampleable,
            TensorConvertible,
        },
        RunMode,
    },
    anyhow::Result,
    candle_core::{
        Device,
        Tensor,
    },
    rand::Rng,
    tracing::warn,
};


pub fn training_loop_off_policy<Alg, Env, Obs, Act>(
    env: &mut Env,
    agent: &mut Alg,
    config: Alg::Config,
    device: &Device,
) -> Result<(Vec<f64>, Vec<bool>)>
where
    Env: Environment<Action = Act, Observation = Obs>,
    Alg: Algorithm + OffPolicyAlgorithm,
    Alg::Config: AlgorithmConfig,
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
            let observation = env.current_observation();
            let state = &<Obs>::to_tensor(observation, device)?;

            // select an action, or randomly sample one
            let action = &if steps_taken < config.initial_random_actions() {
                <Act>::to_tensor(<Act>::sample(&mut rng, &env.action_domain()), device)?
            } else {
                agent.actions(state)?
            };

            let step = env.step(<Act>::from_tensor_pp(action.clone()))?;
            total_reward += step.reward;
            steps_taken += 1;

            agent.remember(
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

        if let RunMode::Train = agent.run_mode() {
            for _ in 0..config.training_iterations() {
                agent.train()?;
            }
        }
    }
    Ok((mc_returns, successes))
}