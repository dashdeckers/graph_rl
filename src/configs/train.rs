use {
    crate::engines::RunMode,
    super::RenderableConfig,
    egui::{
        Ui,
        Label,
        Slider,
    },
    serde::{
        Serialize,
        Deserialize,
    },
};


#[allow(non_camel_case_types)]
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    // The total number of episodes.
    max_episodes: usize,
    // The number of training iterations after one episode finishes.
    training_iterations: usize,
    // Number of random actions to take at very beginning of training.
    initial_random_actions: usize,
    // The RunMode
    run_mode: RunMode,
}
impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            max_episodes: 500,
            training_iterations: 30,
            initial_random_actions: 500,
            run_mode: RunMode::Train,
        }
    }
}
impl TrainConfig {
    pub fn new(
        max_episodes: usize,
        training_iterations: usize,
        initial_random_actions: usize,
        run_mode: RunMode,
    ) -> Self {
        Self {
            max_episodes,
            training_iterations,
            initial_random_actions,
            run_mode,
        }
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
    pub fn run_mode(&self) -> RunMode {
        self.run_mode.clone()
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
    pub fn set_run_mode(&mut self, run_mode: RunMode) {
        self.run_mode = run_mode;
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
            Slider::new(&mut self.training_iterations, 0..=1000)
                .text("Training Iterations")
        );
        ui.add(
            Slider::new(&mut self.initial_random_actions, 0..=1000)
                .text("Initial Random Actions")
        );
        ui.radio_value(&mut self.run_mode, RunMode::Train, "Train");
    }

    fn render_immutable(
        &self,
        ui: &mut Ui,
    ) {
        let max_episodes = self.max_episodes;
        let training_iterations = self.training_iterations;
        let initial_random_actions = self.initial_random_actions;
        let run_mode = self.run_mode.clone();

        ui.separator();
        ui.label("Run Options (Train)");
        ui.add(Label::new(format!("Max Episodes: {max_episodes}")));
        ui.add(Label::new(format!("Training Iterations: {training_iterations}")));
        ui.add(Label::new(format!("Initial Random Actions: {initial_random_actions}")));
        ui.add(Label::new(format!("Run Mode: {run_mode:?}")));
    }
}