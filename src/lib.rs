pub mod util;

pub mod envs;

pub mod ddpg;
pub mod ddpg_sgm;
pub mod ou_noise;
pub mod replay_buffer;
pub mod sgm;

mod config;
pub use config::TrainingConfig;
pub mod gui;

use {
    crate::{
        ddpg::DDPG,
        envs::{
            DistanceMeasure,
            Environment,
            Sampleable,
            TensorConvertible,
            VectorConvertible,
        },
    },
    anyhow::Result,
    candle_core::{
        Device,
        Tensor,
    },
    rand::{
        thread_rng,
        Rng,
    },
    std::{
        fmt::Debug,
        hash::Hash,
    },
    tracing::warn,
};

#[derive(Clone, Copy)]
pub enum RunMode {
    Train,
    Test,
}

pub fn train<E, O, A>(
    env: &mut E,
    agent: &mut DDPG,
    config: TrainingConfig,
    device: &Device,
) -> Result<(Vec<f64>, Vec<bool>)>
where
    E: Environment<Action = A, Observation = O>,
    O: Debug + Clone + Eq + Hash + TensorConvertible + DistanceMeasure,
    A: Clone + VectorConvertible + Sampleable,
{
    warn!("action space: {:?}", env.action_space());
    warn!("observation space: {:?}", env.observation_space());

    let mut steps_taken = 0;
    let mut mc_returns = Vec::new();
    let mut successes = Vec::new();
    let mut rng = rand::thread_rng();

    for episode in 0..config.max_episodes {
        let mut total_reward = 0.0;
        env.reset(rng.gen::<u64>())?;

        for _ in 0..config.episode_length {
            let observation = env.current_observation();
            let state = &<O>::to_tensor(observation, device)?;

            // select an action, or randomly sample one
            let action = if steps_taken < config.initial_random_actions {
                <A>::to_vec(<A>::sample(&mut rng, &env.action_domain()))
            } else {
                agent.actions(state)?
            };

            let step = env.step(<A>::from_vec(action.clone()))?;
            total_reward += step.reward;
            steps_taken += 1;

            agent.remember(
                state,
                &Tensor::new(action, device)?,
                &Tensor::new(vec![step.reward], device)?,
                &<O>::to_tensor(step.observation, device)?,
                step.terminated,
                step.truncated,
            );

            if step.terminated || step.truncated {
                successes.push(step.terminated);
                break;
            }
        }

        warn!("episode {episode} with total reward of {total_reward}");
        mc_returns.push(total_reward);

        if let RunMode::Train = agent.run_mode {
            for _ in 0..config.training_iterations {
                agent.train(config.training_batch_size)?;
            }
        }
    }
    env.reset(rng.gen::<u64>())?;
    Ok((mc_returns, successes))
}

pub fn tick<E, O, A>(
    env: &mut E,
    agent: &mut DDPG,
    device: &Device,
) -> Result<()>
where
    E: Environment<Action = A, Observation = O>,
    O: Clone + TensorConvertible,
    A: Clone + VectorConvertible,
{
    let action = agent.actions(&<O>::to_tensor(env.current_observation(), device)?)?;
    let step = if let Ok(step) = env.step(<A>::from_vec(action.clone())) {
        step
    } else {
        env.reset(thread_rng().gen::<u64>())?;
        env.step(<A>::from_vec(action))?
    };

    if step.terminated || step.truncated {
        env.reset(thread_rng().gen::<u64>())?;
    }
    Ok(())
}
