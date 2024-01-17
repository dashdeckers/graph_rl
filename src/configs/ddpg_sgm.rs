use serde::Serialize;
use {
    super::{
        ActorCriticConfig,
        OffPolicyConfig,
        RenderableConfig,
        DDPG_Config,
        SgmConfig,
        DistanceMode,
    },
    egui::{
        Ui,
        Label,
        Slider,
        Button,
    },
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
    pub fn new(
        ddpg: DDPG_Config,
        distance_mode: DistanceMode,
        sgm_close_enough: f64,
        sgm_maxdist: f64,
        sgm_tau: f64,
    ) -> Self {
        Self {
            ddpg,
            distance_mode,
            sgm_close_enough,
            sgm_maxdist,
            sgm_tau,
        }
    }

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

impl RenderableConfig for DDPG_SGM_Config {
    fn render_immutable(
        &self,
        ui: &mut Ui,
    ) {
        self.ddpg.render_immutable(ui);

        let close_enough = self.sgm_close_enough;
        let maxdist = self.sgm_maxdist;
        let tau = self.sgm_tau;
        let dist_mode = self.distance_mode;

        ui.separator();
        ui.label("SGM Options");
        ui.add(Label::new(format!("Close enough: {close_enough:#.2}")));
        ui.add(Label::new(format!("Max distance: {maxdist:#.2}")));
        ui.add(Label::new(format!("Tau: {tau:#.2}")));
        ui.add(Label::new(format!("Distance mode: {dist_mode}")));
    }

    fn render_mutable(
        &mut self,
        ui: &mut Ui,
    ) {
        self.ddpg.render_mutable(ui);

        ui.separator();
        ui.label("SGM Options");
        ui.add(
            Slider::new(&mut self.sgm_close_enough, 0.0..=1.0)
                .step_by(0.01)
                .text("Close enough"),
        );
        ui.add(
            Slider::new(&mut self.sgm_maxdist, 0.0..=1.0)
                .step_by(0.01)
                .text("Max distance"),
        );
        ui.add(
            Slider::new(&mut self.sgm_tau, 0.0..=1.0)
                .step_by(0.01)
                .text("Tau"),
        );

        let sgm_dist_mode = self.sgm_dist_mode();
        if ui
            .add(Button::new(format!("Toggle DistMode ({sgm_dist_mode})")))
            .clicked()
        {
            self.set_sgm_dist_mode(match self.sgm_dist_mode() {
                DistanceMode::True => DistanceMode::Estimated,
                DistanceMode::Estimated => DistanceMode::True,
            });
        };
    }
}
