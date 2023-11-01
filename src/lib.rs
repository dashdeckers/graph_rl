pub mod logging;

pub mod envs;

pub mod replay_buffer;
pub mod sgm;
pub mod ou_noise;
pub mod ddpg;

pub mod gui;


use envs::Environment;
use ddpg::DDPG;
use anyhow::Result;
use rand::Rng;
use candle_core::{Tensor, Device};


#[derive(Clone)]
pub struct TrainingConfig {
    // The learning rates for the Actor and Critic networks
    pub actor_learning_rate: f64,
    pub critic_learning_rate: f64,
    // The impact of the q value of the next state on the current state's q value.
    pub gamma: f64,
    // The weight for updating the target networks.
    pub tau: f64,
    // The capacity of the replay buffer used for sampling training data.
    pub replay_buffer_capacity: usize,
    // The training batch size for each training iteration.
    pub training_batch_size: usize,
    // The total number of episodes.
    pub max_episodes: usize,
    // The maximum length of an episode.
    pub episode_length: usize,
    // The number of training iterations after one episode finishes.
    pub training_iterations: usize,
    // Ornstein-Uhlenbeck process parameters.
    pub ou_mu: f64,
    pub ou_theta: f64,
    pub ou_sigma: f64,
    // Any postprocessing applied to the actions before they are passed to the environment
    pub postprocess_action: fn(&[f64]) -> Vec<f64>,
    // Any preprocessing applied to the state or goal before they are passed to the agent.
    // This function is used to combine the state and goal in goal conditioned learning.
    pub preprocess_state_goal: fn(&[f64], &[f64]) -> Vec<f64>
}
impl TrainingConfig {
    pub fn pendulum() -> Self {
        Self {
            actor_learning_rate: 1e-4,
            critic_learning_rate: 1e-3,
            gamma: 0.99,
            tau: 0.005,
            replay_buffer_capacity: 100_000,
            training_batch_size: 100,
            max_episodes: 20,
            episode_length: 200,
            training_iterations: 200,
            ou_mu: 0.0,
            ou_theta: 0.15,
            ou_sigma: 0.1,
            postprocess_action: |actions: &[f64]| actions
                .iter()
                .map(|a| (a * 2.0).clamp(-2.0, 2.0))
                .collect(),
            preprocess_state_goal: |state: &[f64], _: &[f64]| state.to_vec(),
        }
    }

    pub fn pointenv(timelimit: usize) -> Self {
        Self {
            actor_learning_rate: 1e-4,
            critic_learning_rate: 1e-3,
            gamma: 0.99,
            tau: 0.005,
            replay_buffer_capacity: 1000,
            training_batch_size: 64,
            max_episodes: 50,
            episode_length: timelimit,
            training_iterations: 200,
            ou_mu: 0.0,
            ou_theta: 0.15,
            ou_sigma: 0.1,
            postprocess_action: |actions: &[f64]| actions.to_vec(),
            preprocess_state_goal: |state: &[f64], goal: &[f64]| [state, goal].concat(),
        }
    }

    pub fn gui(
        timelimit: usize,
        n_episodes: usize,
    ) -> Self {
        Self {
            actor_learning_rate: 1e-4,
            critic_learning_rate: 1e-3,
            gamma: 0.99,
            tau: 0.005,
            replay_buffer_capacity: 1000,
            training_batch_size: 64,
            max_episodes: n_episodes,
            episode_length: timelimit,
            training_iterations: 0,
            ou_mu: 0.0,
            ou_theta: 0.15,
            ou_sigma: 0.1,
            postprocess_action: |actions: &[f64]| actions.to_vec(),
            preprocess_state_goal: |state: &[f64], goal: &[f64]| [state, goal].concat(),
        }
    }
}


#[allow(clippy::too_many_arguments)]
pub fn run<E: Environment>(
    env: &mut E,
    agent: &mut DDPG,
    config: TrainingConfig,
    train: bool,
    device: &Device,
) -> Result<()> {

    println!("action space: {}", env.action_space());
    println!("observation space: {:?}", env.observation_space());

    let mut rng = rand::thread_rng();

    agent.train = train;
    for episode in 0..config.max_episodes {
        // let mut state = env.reset(episode as u64)?;
        let mut state = env.reset(rng.gen::<u64>())?;

        let mut total_reward = 0.0;
        for _ in 0..config.episode_length {
            let obs: Vec<f64> = state.clone().into();
            let goal: Vec<f64> = env.current_goal().into();

            let action = agent.actions(&Tensor::new((config.preprocess_state_goal)(&obs, &goal), device)?)?;
            let action = (config.postprocess_action)(&action);

            let step = env.step(action.clone().into())?;
            let next_obs: Vec<f64> = step.state.clone().into();

            total_reward += step.reward;
            if train {
                let state = &Tensor::new((config.preprocess_state_goal)(&obs, &goal), device)?;
                let action = &Tensor::new(action, device)?;
                let reward = &Tensor::new(vec![step.reward], device)?;
                let next_state = &Tensor::new((config.preprocess_state_goal)(&next_obs, &goal), device)?;

                agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    step.terminated,
                    step.truncated,
                );
            }

            if step.terminated || step.truncated {
                break;
            }
            state = step.state;
        }

        println!("episode {episode} with total reward of {total_reward}");

        if train {
            for _ in 0..config.training_iterations {
                agent.train(config.training_batch_size)?;
            }
        }
    }
    env.reset(rng.gen::<u64>())?;
    Ok(())
}


pub fn tick<E: Environment>(
    env: &mut E,
    agent: &mut DDPG,
    config: TrainingConfig,
    train: bool,
    device: &Device,
) -> Result<()> {

    agent.train = train;

    let obs: Vec<f64> = env.current_state().into();
    let goal: Vec<f64> = env.current_goal().into();

    let action = agent.actions(&Tensor::new((config.preprocess_state_goal)(&obs, &goal), device)?)?;
    let action = (config.postprocess_action)(&action);

    let step = env.step(action.clone().into())?;

    if step.terminated || step.truncated {
        env.reset(rand::thread_rng().gen::<u64>())?;
    }

    Ok(())
}