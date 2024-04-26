use {
    super::RenderableConfig,
    serde::{
        Serialize,
        Deserialize,
    },
    egui::{
        Ui,
        Label,
        Slider,
    },
};


#[allow(non_camel_case_types)]
#[derive(Clone, Serialize, Deserialize)]
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
impl Default for DDPG_Config {
    fn default() -> Self {
        Self {
            actor_learning_rate: 0.0003,
            critic_learning_rate: 0.0003,
            gamma: 0.99,
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
        let hidden_1_size = self.hidden_1_size;
        let hidden_2_size = self.hidden_2_size;
        let buffer_size = self.replay_buffer_capacity;
        let batch_size = self.training_batch_size;
        let ou_kappa = self.ou_kappa;
        let ou_sigma = self.ou_sigma;

        ui.separator();
        ui.label("DDPG Options");
        ui.add(Label::new(format!("Actor LR: {actor_lr:#.5}")));
        ui.add(Label::new(format!("Critic LR: {critic_lr:#.5}")));
        ui.add(Label::new(format!("Gamma: {gamma}")));
        ui.add(Label::new(format!("Tau: {tau}")));
        ui.add(Label::new(format!("Hidden 1 size: {hidden_1_size}")));
        ui.add(Label::new(format!("Hidden 2 size: {hidden_2_size}")));
        ui.add(Label::new(format!("Buffer size: {buffer_size}")));
        ui.add(Label::new(format!("Batch size: {batch_size}")));
        ui.add(Label::new(format!("OU Kappa (speed): {ou_kappa}")));
        ui.add(Label::new(format!("OU Sigma (volatility): {ou_sigma}")));
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
            Slider::new(&mut self.hidden_1_size, 0..=1_000)
                .text("Hidden 1 size"),
        );
        ui.add(
            Slider::new(&mut self.hidden_2_size, 0..=1_000)
                .text("Hidden 2 size"),
        );
        ui.add(
            Slider::new(&mut self.replay_buffer_capacity, 0..=1_000_000)
                .text("Buffer size"),
        );
        ui.add(
            Slider::new(&mut self.training_batch_size, 0..=1_000)
                .text("Batch size"),
        );
        ui.add(
            Slider::new(&mut self.ou_kappa, 0.0..=1.0)
                .step_by(0.001)
                .text("OU Kappa (speed)"),
        );
        ui.add(
            Slider::new(&mut self.ou_sigma, 0.0..=1.0)
                .step_by(0.001)
                .text("OU Sigma (volatility)"),
        );
    }
}
