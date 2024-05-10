use {
    super::{
        RenderableConfig,
        DDPG_Config,
        DistanceMode,
    },
    serde::{
        Serialize,
        Deserialize,
    },
    egui::{
        Ui,
        Label,
        Slider,
        Button,
    },
};


#[allow(non_camel_case_types)]
#[derive(Clone, Serialize, Deserialize)]
pub struct DDPG_HGB_Config {
    // The base DDPG parameters
    pub ddpg: DDPG_Config,
    // Whether to use true or estimated distances
    pub distance_mode: DistanceMode,
    // Sparse Graphical Memory parameters
    pub sgm_reconstruct_freq: usize,
    pub sgm_max_tries: usize,
    pub sgm_close_enough: f64,
    pub sgm_waypoint_reward: f64,
    pub sgm_maxdist: f64,
    pub sgm_tau: f64,
}
impl Default for DDPG_HGB_Config {
    fn default() -> Self {
        Self {
            ddpg: DDPG_Config::default(),
            distance_mode: DistanceMode::True,
            sgm_reconstruct_freq: 50,
            sgm_max_tries: 5,
            sgm_close_enough: 0.5,
            sgm_waypoint_reward: 1.0,
            sgm_maxdist: 1.0,
            sgm_tau: 0.4,
        }
    }
}
impl DDPG_HGB_Config {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ddpg: DDPG_Config,
        distance_mode: DistanceMode,
        sgm_reconstruct_freq: usize,
        sgm_max_tries: usize,
        sgm_close_enough: f64,
        sgm_waypoint_reward: f64,
        sgm_maxdist: f64,
        sgm_tau: f64,
    ) -> Self {
        Self {
            ddpg,
            distance_mode,
            sgm_reconstruct_freq,
            sgm_max_tries,
            sgm_close_enough,
            sgm_waypoint_reward,
            sgm_maxdist,
            sgm_tau,
        }
    }
}

impl RenderableConfig for DDPG_HGB_Config {
    fn render_immutable(
        &self,
        ui: &mut Ui,
    ) {
        self.ddpg.render_immutable(ui);

        let dist_mode = self.distance_mode;
        let sgm_reconstruct_freq = self.sgm_reconstruct_freq;
        let sgm_max_tries = self.sgm_max_tries;
        let close_enough = self.sgm_close_enough;
        let waypoint_reward = self.sgm_waypoint_reward;
        let maxdist = self.sgm_maxdist;
        let tau = self.sgm_tau;

        ui.separator();
        ui.label("SGM Options");
        ui.add(Label::new(format!("Distance mode: {dist_mode}")));
        ui.add(Label::new(format!("Reconstruct freq: {sgm_reconstruct_freq}")));
        ui.add(Label::new(format!("Max tries: {sgm_max_tries}")));
        ui.add(Label::new(format!("Close enough: {close_enough:#.2}")));
        ui.add(Label::new(format!("Waypoint reward: {waypoint_reward:#.2}")));
        ui.add(Label::new(format!("Max distance: {maxdist:#.2}")));
        ui.add(Label::new(format!("Tau: {tau:#.2}")));
    }

    fn render_mutable(
        &mut self,
        ui: &mut Ui,
    ) {
        self.ddpg.render_mutable(ui);

        ui.separator();
        ui.label("SGM Options");
        let distance_mode = self.distance_mode;
        if ui
            .add(Button::new(format!("Toggle DistMode ({distance_mode})")))
            .clicked()
        {
            self.distance_mode = match distance_mode {
                DistanceMode::True => DistanceMode::Estimated,
                DistanceMode::Estimated => DistanceMode::True,
            };
        };
        ui.add(
            Slider::new(&mut self.sgm_reconstruct_freq, 0..=100)
                .text("Reconstruct freq"),
        );
        ui.add(
            Slider::new(&mut self.sgm_max_tries, 1..=50)
                .text("Max tries"),
        );
        ui.add(
            Slider::new(&mut self.sgm_close_enough, 0.0..=1.0)
                .step_by(0.01)
                .text("Close enough"),
        );
        ui.add(
            Slider::new(&mut self.sgm_waypoint_reward, 0.0..=10.0)
                .step_by(0.1)
                .text("Waypoint reward"),
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

    }
}
