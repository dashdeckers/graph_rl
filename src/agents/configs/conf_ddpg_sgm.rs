use serde::Serialize;
use super::{
    DDPG_Config,
    AlgorithmConfig,
    ActorCriticConfig,
    OffPolicyConfig,
    SgmConfig,
    DistanceMode,
};


#[allow(non_camel_case_types)]
#[derive(Clone, Serialize)]
pub struct DDPG_SGM_Config {
    // The base DDPG parameters
    pub ddpg: DDPG_Config,
    // Whether to use true or estimated distances
    pub distance_mode: DistanceMode,
    // Sparse Graphical Memory parameters
    pub sgm_close_enough: f64,
    pub sgm_maxdist: f64,
    pub sgm_tau: f64,
}
impl DDPG_SGM_Config {
    pub fn pendulum() -> Self {
        Self {
            ddpg: DDPG_Config::pendulum(),
            distance_mode: DistanceMode::True,
            sgm_close_enough: 0.5,
            sgm_maxdist: 1.0,
            sgm_tau: 0.4,
        }
    }

    pub fn pointenv() -> Self {
        Self {
            ddpg: DDPG_Config::pointenv(),
            distance_mode: DistanceMode::True,
            sgm_close_enough: 0.5,
            sgm_maxdist: 1.0,
            sgm_tau: 0.4,
        }
    }

    pub fn pointmaze() -> Self {
        Self {
            ddpg: DDPG_Config::pointmaze(),
            distance_mode: DistanceMode::True,
            sgm_close_enough: 0.5,
            sgm_maxdist: 1.0,
            sgm_tau: 0.4,
        }
    }
}


impl AlgorithmConfig for DDPG_SGM_Config {
    fn max_episodes(&self) -> usize {
        self.ddpg.max_episodes
    }
    fn training_iterations(&self) -> usize {
        self.ddpg.training_iterations
    }
    fn initial_random_actions(&self) -> usize {
        self.ddpg.initial_random_actions
    }
    fn set_max_episodes(&mut self, max_episodes: usize) {
        self.ddpg.max_episodes = max_episodes;
    }
    fn set_training_iterations(&mut self, training_iterations: usize) {
        self.ddpg.training_iterations = training_iterations;
    }
    fn set_initial_random_actions(&mut self, initial_random_actions: usize) {
        self.ddpg.initial_random_actions = initial_random_actions;
    }
}

impl ActorCriticConfig for DDPG_SGM_Config {
    fn actor_lr(&self) -> f64 {
        self.ddpg.actor_learning_rate
    }
    fn critic_lr(&self) -> f64 {
        self.ddpg.critic_learning_rate
    }
    fn gamma(&self) -> f64 {
        self.ddpg.gamma
    }
    fn tau(&self) -> f64 {
        self.ddpg.tau
    }
    fn set_actor_lr(&mut self, lr: f64) {
        self.ddpg.actor_learning_rate = lr;
    }
    fn set_critic_lr(&mut self, lr: f64) {
        self.ddpg.critic_learning_rate = lr;
    }
    fn set_gamma(&mut self, gamma: f64) {
        self.ddpg.gamma = gamma;
    }
    fn set_tau(&mut self, tau: f64) {
        self.ddpg.tau = tau;
    }
}

impl OffPolicyConfig for DDPG_SGM_Config {
    fn replay_buffer_capacity(&self) -> usize {
        self.ddpg.replay_buffer_capacity
    }
    fn training_batch_size(&self) -> usize {
        self.ddpg.training_batch_size
    }
    fn set_replay_buffer_capacity(&mut self, capacity: usize) {
        self.ddpg.replay_buffer_capacity = capacity;
    }
    fn set_training_batch_size(&mut self, batch_size: usize) {
        self.ddpg.training_batch_size = batch_size;
    }
}

impl SgmConfig for DDPG_SGM_Config {
    fn sgm_close_enough(&self) -> f64 {
        self.sgm_close_enough
    }
    fn sgm_maxdist(&self) -> f64 {
        self.sgm_maxdist
    }
    fn sgm_tau(&self) -> f64 {
        self.sgm_tau
    }
    fn sgm_dist_mode(&self) -> DistanceMode {
        self.distance_mode
    }
    fn set_sgm_close_enough(&mut self, sgm_close_enough: f64) {
        self.sgm_close_enough = sgm_close_enough;
    }
    fn set_sgm_maxdist(&mut self, maxdist: f64) {
        self.sgm_maxdist = maxdist;
    }
    fn set_sgm_tau(&mut self, tau: f64) {
        self.sgm_tau = tau;
    }
    fn set_sgm_dist_mode(&mut self, dist_mode: DistanceMode) {
        self.distance_mode = dist_mode;
    }
}