use serde::Serialize;
use super::{
    AlgorithmConfig,
    ActorCriticConfig,
    OffPolicyConfig,
};


#[allow(non_camel_case_types)]
#[derive(Clone, Serialize)]
pub struct DDPG_Config {
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
    // The number of training iterations after one episode finishes.
    pub training_iterations: usize,
    // Number of random actions to take at very beginning of training.
    pub initial_random_actions: usize,
    // Ornstein-Uhlenbeck process parameters.
    pub ou_mu: f64,
    pub ou_theta: f64,
    pub ou_sigma: f64,
}
impl DDPG_Config {
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
            max_episodes: 30,
            training_iterations: 200,
            initial_random_actions: 0,
            ou_mu: 0.0,
            ou_theta: 0.15,
            ou_sigma: 0.1,
        }
    }

    pub fn pointenv() -> Self {
        Self {
            actor_learning_rate: 1e-4, // 0.0003
            critic_learning_rate: 1e-3, // 0.0003
            gamma: 1.0, // 0.99
            tau: 0.005,
            hidden_1_size: 400, // 256
            hidden_2_size: 300, // 256
            replay_buffer_capacity: 10_000,
            training_batch_size: 256,
            max_episodes: 50,
            training_iterations: 100,
            initial_random_actions: 1000,
            ou_mu: 0.0,
            ou_theta: 0.15,
            ou_sigma: 0.2,
        }
    }

    pub fn pointmaze() -> Self {
        Self::pointenv()
    }
}


impl AlgorithmConfig for DDPG_Config {
    fn max_episodes(&self) -> usize {
        self.max_episodes
    }
    fn training_iterations(&self) -> usize {
        self.training_iterations
    }
    fn initial_random_actions(&self) -> usize {
        self.initial_random_actions
    }
    fn set_max_episodes(&mut self, max_episodes: usize) {
        self.max_episodes = max_episodes;
    }
    fn set_training_iterations(&mut self, training_iterations: usize) {
        self.training_iterations = training_iterations;
    }
    fn set_initial_random_actions(&mut self, initial_random_actions: usize) {
        self.initial_random_actions = initial_random_actions;
    }
}

impl ActorCriticConfig for DDPG_Config {
    fn actor_lr(&self) -> f64 {
        self.actor_learning_rate
    }
    fn critic_lr(&self) -> f64 {
        self.critic_learning_rate
    }
    fn gamma(&self) -> f64 {
        self.gamma
    }
    fn tau(&self) -> f64 {
        self.tau
    }
    fn set_actor_lr(&mut self, lr: f64) {
        self.actor_learning_rate = lr;
    }
    fn set_critic_lr(&mut self, lr: f64) {
        self.critic_learning_rate = lr;
    }
    fn set_gamma(&mut self, gamma: f64) {
        self.gamma = gamma;
    }
    fn set_tau(&mut self, tau: f64) {
        self.tau = tau;
    }
}

impl OffPolicyConfig for DDPG_Config {
    fn replay_buffer_capacity(&self) -> usize {
        self.replay_buffer_capacity
    }
    fn training_batch_size(&self) -> usize {
        self.training_batch_size
    }
    fn set_replay_buffer_capacity(&mut self, capacity: usize) {
        self.replay_buffer_capacity = capacity;
    }
    fn set_training_batch_size(&mut self, batch_size: usize) {
        self.training_batch_size = batch_size;
    }
}
