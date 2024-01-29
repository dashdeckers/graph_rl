use {
    super::RenderableConfig,
    egui::{
        Ui,
        Label,
        Slider,
    },
    serde::Serialize,
};

#[allow(non_camel_case_types)]
#[derive(Clone, Serialize)]
pub struct TestConfig {
    // The total number of episodes.
    max_episodes: usize,
}
impl TestConfig {
    pub fn new(max_episodes: usize) -> Self {
        Self {
            max_episodes,
        }
    }

    pub fn pendulum() -> Self {
        Self {
            max_episodes: 30,
        }
    }

    pub fn pointenv() -> Self {
        Self {
            max_episodes: 300,
        }
    }

    pub fn pointmaze() -> Self {
        Self::pointenv()
    }
}

impl TestConfig {
    pub fn max_episodes(&self) -> usize {
        self.max_episodes
    }
    pub fn set_max_episodes(&mut self, max_episodes: usize) {
        self.max_episodes = max_episodes;
    }
}

impl RenderableConfig for TestConfig {
    fn render_mutable(
        &mut self,
        ui: &mut Ui,
    ) {
        ui.separator();
        ui.label("Run Options (Test)");
        ui.add(
            Slider::new(&mut self.max_episodes, 1..=1000)
                .text("Max Episodes")
        );
    }

    fn render_immutable(
        &self,
        ui: &mut Ui,
    ) {
        let max_episodes = self.max_episodes;

        ui.separator();
        ui.label("Run Options (Test)");
        ui.add(Label::new(format!("Max Episodes: {max_episodes}")));
    }
}
