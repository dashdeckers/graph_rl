use serde::Serialize;
use {
    super::{
        ActorCriticConfig,
        OffPolicyConfig,
        RenderableConfig,
    },
    egui::{
        Ui,
        Label,
        Slider,
    },
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
    // Ornstein-Uhlenbeck process parameters.
    pub ou_theta: f64,
    pub ou_kappa: f64,
    pub ou_sigma: f64,
}
impl DDPG_Config {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        actor_learning_rate: f64,
        critic_learning_rate: f64,
        gamma: f64,
        tau: f64,
        hidden_1_size: usize,
        hidden_2_size: usize,
        replay_buffer_capacity: usize,
        training_batch_size: usize,
        ou_theta: f64,
        ou_kappa: f64,
        ou_sigma: f64,
    ) -> Self {
        Self {
            actor_learning_rate,
            critic_learning_rate,
            gamma,
            tau,
            hidden_1_size,
            hidden_2_size,
            replay_buffer_capacity,
            training_batch_size,
            ou_theta,
            ou_kappa,
            ou_sigma,
        }
    }

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
            ou_theta: 0.0,
            ou_kappa: 0.15,
            ou_sigma: 0.1,
        }
    }

    pub fn pointenv() -> Self {
        Self {
            actor_learning_rate: 0.0003,
            critic_learning_rate: 0.0003,
            gamma: 1.0,
            tau: 0.005,
            hidden_1_size: 256,
            hidden_2_size: 256,
            replay_buffer_capacity: 1_000,
            training_batch_size: 64,
            ou_theta: 0.0,
            ou_kappa: 0.15,
            ou_sigma: 0.2,
        }
    }

    pub fn pointmaze() -> Self {
        Self::pointenv()
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

impl RenderableConfig for DDPG_Config {
    fn render_immutable(
        &self,
        ui: &mut Ui,
    ) {
        let actor_lr = self.actor_learning_rate;
        let critic_lr = self.critic_learning_rate;
        let gamma = self.gamma;
        let tau = self.tau;
        let buffer_size = self.replay_buffer_capacity;
        let batch_size = self.training_batch_size;

        ui.separator();
        ui.label("DDPG Options");
        ui.add(Label::new(format!("Actor LR: {actor_lr:#.5}")));
        ui.add(Label::new(format!("Critic LR: {critic_lr:#.5}")));
        ui.add(Label::new(format!("Gamma: {gamma}")));
        ui.add(Label::new(format!("Tau: {tau}")));
        ui.add(Label::new(format!("Buffer size: {buffer_size}")));
        ui.add(Label::new(format!("Batch size: {batch_size}")));
    }

    fn render_mutable(
        &mut self,
        ui: &mut Ui,
    ) {
        ui.separator();
        ui.label("DDPG Options");
        ui.add(
            Slider::new(&mut self.actor_learning_rate, 0.0..=1.0)
                .step_by(0.0001)
                .text("Actor LR"),
        );
        ui.add(
            Slider::new(&mut self.critic_learning_rate, 0.0..=1.0)
                .step_by(0.0001)
                .text("Critic LR"),
        );
        ui.add(
            Slider::new(&mut self.gamma, 0.0..=1.0)
                .step_by(0.0001)
                .text("Gamma"),
        );
        ui.add(
            Slider::new(&mut self.tau, 0.0..=1.0)
                .step_by(0.0001)
                .text("Tau"),
        );
        ui.add(
            Slider::new(&mut self.replay_buffer_capacity, 0..=1_000_000)
                .text("Buffer size"),
        );
        ui.add(
            Slider::new(&mut self.training_batch_size, 0..=1_000)
                .text("Batch size"),
        );
    }
}
