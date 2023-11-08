pub mod util;

pub mod envs;

pub mod replay_buffer;
pub mod sgm;
pub mod ou_noise;
pub mod ddpg;
pub mod ddpg_sgm;

pub mod gui;


use std::hash::Hash;
use std::fmt::Debug;

use anyhow::Result;
use tracing::warn;
use rand::{Rng, thread_rng};
use candle_core::{Device, Tensor};
use crate::{
    ddpg::DDPG,
    envs::{
        Environment,
        VectorConvertible,
        TensorConvertible,
        DistanceMeasure,
    },
};


#[derive(Clone)]
pub struct TrainingConfig {
    // The learning rates for the Actor and Critic networks
    pub actor_learning_rate: f64,
    pub critic_learning_rate: f64,
    // The impact of the q value of the next state on the current state's q value.
    pub gamma: f64,
    // The weight for updating the target networks.
    pub tau: f64,
    // The number of neurons in the hidden layers of the Actor and Critic networks.
    pub hidden_1_size: usize,
    pub hidden_2_size: usize,
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
    // Sparse Graphical Memory parameters
    pub sgm_freq: usize,
    pub sgm_maxdist: f64,
    pub sgm_tau: f64,
}
impl TrainingConfig {
    pub fn pendulum() -> Self {
        Self {
            actor_learning_rate: 1e-4,
            critic_learning_rate: 1e-3,
            gamma: 0.99,
            tau: 0.005,
            hidden_1_size: 400,
            hidden_2_size: 300,
            replay_buffer_capacity: 100_000,
            training_batch_size: 100,
            max_episodes: 100,
            episode_length: 200,
            training_iterations: 200,
            ou_mu: 0.0,
            ou_theta: 0.15,
            ou_sigma: 0.1,
            sgm_freq: 1,
            sgm_maxdist: 1.0,
            sgm_tau: 0.4,
        }
    }

    pub fn pointenv(timelimit: usize) -> Self {
        Self {
            actor_learning_rate: 1e-4,
            critic_learning_rate: 1e-3,
            gamma: 0.99,
            tau: 0.005,
            hidden_1_size: 512,
            hidden_2_size: 512,
            replay_buffer_capacity: 1000,
            training_batch_size: 100,
            max_episodes: 50,
            episode_length: timelimit,
            training_iterations: 200,
            ou_mu: 0.0,
            ou_theta: 0.15,
            ou_sigma: 0.1,
            sgm_freq: 1,
            sgm_maxdist: 1.0,
            sgm_tau: 0.4,
        }
    }
}


pub fn run<E, O, A>(
    env: &mut E,
    agent: &mut DDPG,
    config: TrainingConfig,
    train: bool,
    device: &Device,
) -> Result<Vec<f64>>
where
    E: Environment<Action = A, Observation = O>,
    O: Debug + Clone + Eq + Hash + TensorConvertible + DistanceMeasure,
    A: Clone + VectorConvertible,
{
    warn!("action space: {:?}", env.action_space());
    warn!("observation space: {:?}", env.observation_space());

    let mut episodic_returns = Vec::new();
    let mut rng = rand::thread_rng();

    agent.train = train;
    for episode in 0..config.max_episodes {
        let mut total_reward = 0.0;
        env.reset(rng.gen::<u64>())?;

        for _ in 0..config.episode_length {
            let observation = env.current_observation();
            let state = &<O>::to_tensor(observation, device)?;

            let action = agent.actions(state)?;
            let step = env.step(<A>::from_vec(action.clone()))?;
            total_reward += step.reward;

            if train {
                agent.remember(
                    state,
                    &Tensor::new(action, device)?,
                    &Tensor::new(vec![step.reward], device)?,
                    &<O>::to_tensor(step.observation, device)?,
                    step.terminated,
                    step.truncated,
                );
            }

            if step.terminated || step.truncated {
                break;
            }
        }

        warn!("episode {episode} with total reward of {total_reward}");
        episodic_returns.push(total_reward);

        if train {
            for _ in 0..config.training_iterations {
                agent.train(config.training_batch_size)?;
            }
        }
    }
    env.reset(rng.gen::<u64>())?;
    Ok(episodic_returns)
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
    let step = env.step(<A>::from_vec(action))?;
    if step.terminated || step.truncated {
        env.reset(thread_rng().gen::<u64>())?;
    }
    Ok(())
}