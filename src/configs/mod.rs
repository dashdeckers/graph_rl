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