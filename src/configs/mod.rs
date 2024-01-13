mod train;
mod test;
mod ddpg;
mod ddpg_sgm;

pub use train::TrainConfig;
pub use test::TestConfig;
pub use ddpg::DDPG_Config;
pub use ddpg_sgm::DDPG_SGM_Config;

use crate::components::sgm::DistanceMode;
use egui::Ui;


pub trait ActorCriticConfig {
    fn actor_lr(&self) -> f64;
    fn critic_lr(&self) -> f64;
    fn gamma(&self) -> f64;
    fn tau(&self) -> f64;
    fn set_actor_lr(&mut self, lr: f64);
    fn set_critic_lr(&mut self, lr: f64);
    fn set_gamma(&mut self, gamma: f64);
    fn set_tau(&mut self, tau: f64);
}

pub trait OffPolicyConfig {
    fn replay_buffer_capacity(&self) -> usize;
    fn training_batch_size(&self) -> usize;
    fn set_replay_buffer_capacity(&mut self, capacity: usize);
    fn set_training_batch_size(&mut self, batch_size: usize);
}

pub trait SgmConfig {
    fn sgm_close_enough(&self) -> f64;
    fn sgm_maxdist(&self) -> f64;
    fn sgm_tau(&self) -> f64;
    fn sgm_dist_mode(&self) -> DistanceMode;
    fn set_sgm_close_enough(&mut self, close_enough: f64);
    fn set_sgm_maxdist(&mut self, maxdist: f64);
    fn set_sgm_tau(&mut self, tau: f64);
    fn set_sgm_dist_mode(&mut self, dist_mode: DistanceMode);
}

pub trait RenderableConfig {
    fn render_mutable(
        &mut self,
        ui: &mut Ui,
    );

    fn render_immutable(
        &self,
        ui: &mut Ui,
    );
}