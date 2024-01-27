use serde::Serialize;
use {
    super::{
        RenderableConfig,
        DDPG_Config,
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
    pub sgm_max_tries: usize,
    pub sgm_close_enough: f64,
    pub sgm_maxdist: f64,
    pub sgm_tau: f64,
}
impl DDPG_SGM_Config {
    pub fn new(
        ddpg: DDPG_Config,
        distance_mode: DistanceMode,
        sgm_max_tries: usize,
        sgm_close_enough: f64,
        sgm_maxdist: f64,
        sgm_tau: f64,
    ) -> Self {
        Self {
            ddpg,
            distance_mode,
            sgm_max_tries,
            sgm_close_enough,
            sgm_maxdist,
            sgm_tau,
        }
    }

    pub fn large() -> Self {
        Self {
            ddpg: DDPG_Config::large(),
            distance_mode: DistanceMode::True,
            sgm_max_tries: 5,
            sgm_close_enough: 0.5,
            sgm_maxdist: 1.0,
            sgm_tau: 0.4,
        }
    }

    pub fn small() -> Self {
        Self {
            ddpg: DDPG_Config::small(),
            ..Self::large()
        }
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
    }
}
