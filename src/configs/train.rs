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
pub struct TrainConfig {
    // The total number of episodes.
    max_episodes: usize,
    // The number of training iterations after one episode finishes.
    training_iterations: usize,
    // Number of random actions to take at very beginning of training.
    initial_random_actions: usize,
}
impl TrainConfig {
    pub fn new(
        max_episodes: usize,
        training_iterations: usize,
        initial_random_actions: usize,
    ) -> Self {
        Self {
            max_episodes,
            training_iterations,
            initial_random_actions,
        }
    }

    pub fn pendulum() -> Self {
        Self {
            max_episodes: 30,
            training_iterations: 200,
            initial_random_actions: 0,
        }
    }

    pub fn pointenv() -> Self {
        Self {
            max_episodes: 300,
            training_iterations: 30,
            initial_random_actions: 1000,
        }
    }

    pub fn pointmaze() -> Self {
        Self::pointenv()
    }
}

impl TrainConfig {
    pub fn max_episodes(&self) -> usize {
        self.max_episodes
    }
    pub fn training_iterations(&self) -> usize {
        self.training_iterations
    }
    pub fn initial_random_actions(&self) -> usize {
        self.initial_random_actions
    }
    pub fn set_max_episodes(&mut self, max_episodes: usize) {
        self.max_episodes = max_episodes;
    }
    pub fn set_training_iterations(&mut self, training_iterations: usize) {
        self.training_iterations = training_iterations;
    }
    pub fn set_initial_random_actions(&mut self, initial_random_actions: usize) {
        self.initial_random_actions = initial_random_actions;
    }
}

impl RenderableConfig for TrainConfig {
    fn render_mutable(
        &mut self,
        ui: &mut Ui,
    ) {
        ui.separator();
        ui.label("Run Options (Train)");
        ui.add(
            Slider::new(&mut self.max_episodes, 1..=1000)
                .text("Max Episodes")
        );
        ui.add(
            Slider::new(&mut self.training_iterations, 1..=1000)
                .text("Training Iterations")
        );
        ui.add(
            Slider::new(&mut self.initial_random_actions, 0..=1000)
                .text("Initial Random Actions")
        );
    }

    fn render_immutable(
        &self,
        ui: &mut Ui,
    ) {
        let max_episodes = self.max_episodes;
        let training_iterations = self.training_iterations;
        let initial_random_actions = self.initial_random_actions;

        ui.separator();
        ui.label("Run Options (Train)");
        ui.add(Label::new(format!("Max Episodes: {max_episodes}")));
        ui.add(Label::new(format!("Training Iterations: {training_iterations}")));
        ui.add(Label::new(format!("Initial Random Actions: {initial_random_actions}")));
    }
}