mod ddpg;
mod ddpg_sgm;

pub use ddpg::DDPG_Config;
pub use ddpg_sgm::DDPG_SGM_Config;


use serde::Serialize;


#[derive(Clone, Serialize)]
pub enum DistanceMode {
    True,
    Estimated,
}

pub trait AlgorithmConfig {
    fn max_episodes(&self) -> usize;
    fn training_iterations(&self) -> usize;
    fn initial_random_actions(&self) -> usize;
    fn set_max_episodes(&mut self, max_episodes: usize);
    fn set_training_iterations(&mut self, training_iterations: usize);
    fn set_initial_random_actions(&mut self, initial_random_actions: usize);
}
pub trait ActorCriticConfig: AlgorithmConfig {
    fn actor_lr(&self) -> f64;
    fn critic_lr(&self) -> f64;
    fn gamma(&self) -> f64;
    fn tau(&self) -> f64;
    fn set_actor_lr(&mut self, lr: f64);
    fn set_critic_lr(&mut self, lr: f64);
    fn set_gamma(&mut self, gamma: f64);
    fn set_tau(&mut self, tau: f64);
}
pub trait OffPolicyConfig: AlgorithmConfig {
    fn replay_buffer_capacity(&self) -> usize;
    fn training_batch_size(&self) -> usize;
    fn set_replay_buffer_capacity(&mut self, capacity: usize);
    fn set_training_batch_size(&mut self, batch_size: usize);
}
pub trait SgmConfig: AlgorithmConfig {
    fn sgm_close_enough(&self) -> f64;
    fn sgm_maxdist(&self) -> f64;
    fn sgm_tau(&self) -> f64;
    fn set_sgm_close_enough(&mut self, close_enough: f64);
    fn set_sgm_maxdist(&mut self, maxdist: f64);
    fn set_sgm_tau(&mut self, tau: f64);
}
