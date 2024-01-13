use {
    super::{
        line::PointLine,
        reward::PointReward,
    },
    crate::configs::RenderableConfig,
    serde::Serialize,
    rand::{
        rngs::StdRng,
        Rng,
        SeedableRng,
    },
    egui::{
        Ui,
        Label,
        Slider,
    },
};

/// The configuration struct for the [`PointEnv`](super::point_env::PointEnv) environment.
///
/// # Fields
/// * `width` - The width of the environment.
/// * `height` - The height of the environment.
/// * `walls` - The walls of the environment given as a Vec of [`PointLines`](super::line::PointLine)
/// * `timelimit` - The maximum number of steps before the episode is truncated.
/// * `step_radius` - The radius that defines the maximum distance the agent can reach in one step.
/// * `term_radius` - If the agent is within this radius of the goal, the episode is terminated.
/// * `max_radius` - The maximum distance allowed between the randomly generated start and goal.
/// * `bounce_factor` - The percentage of the traveled distance that the agent bounces back when it hits a wall.
/// * `reward` - The reward function. For more information, see [`PointReward`](super::reward::PointReward)
/// * `seed` - The seed for the random number generator.
///
/// # Example
/// ```
/// use graph_rl::envs::{
///     PointEnvConfig,
///     PointReward,
/// };
///
/// let config = PointEnvConfig::default();
/// assert_eq!(config.width, 5);
/// assert_eq!(config.height, 5);
/// assert_eq!(config.walls, None);
/// assert_eq!(config.timelimit, 30);
/// assert_eq!(config.step_radius, 1.0);
/// assert_eq!(config.term_radius, 0.5);
/// assert_eq!(config.max_radius, None);
/// assert_eq!(config.bounce_factor, 0.1);
/// assert_eq!(config.reward, PointReward::Distance);
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct PointEnvConfig {
    pub width: f64,
    pub height: f64,
    pub walls: Option<Vec<PointLine>>,
    pub timelimit: usize,
    pub step_radius: f64,
    pub term_radius: f64,
    pub max_radius: Option<f64>,
    pub bounce_factor: f64,
    pub reward: PointReward,
    pub seed: u64,
}
impl Default for PointEnvConfig {
    fn default() -> Self {
        Self {
            width: 5.0,
            height: 5.0,
            walls: None,
            timelimit: 30,
            step_radius: 1.0,
            term_radius: 0.5,
            max_radius: None,
            bounce_factor: 0.1,
            reward: PointReward::Distance,
            seed: StdRng::from_entropy().gen::<u64>(),
        }
    }
}
impl PointEnvConfig {
    /// Creates a new [PointEnvConfig].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: f64,
        height: f64,
        walls: Option<Vec<PointLine>>,
        timelimit: usize,
        step_radius: f64,
        term_radius: f64,
        max_radius: Option<f64>,
        bounce_factor: f64,
        reward: PointReward,
        seed: u64,
    ) -> Self {
        Self {
            width,
            height,
            walls,
            timelimit,
            step_radius,
            term_radius,
            max_radius,
            bounce_factor,
            reward,
            seed,
        }
    }
}

impl RenderableConfig for PointEnvConfig {
    fn render_immutable(
        &self,
        ui: &mut Ui,
    ) {
        let width = self.width;
        let height = self.height;
        let timelimit = self.timelimit;
        let step_radius = self.step_radius;
        let term_radius = self.term_radius;
        let bounce_factor = self.bounce_factor;
        let reward = &self.reward;
        let seed = self.seed;

        ui.label("PointEnv Options");
        ui.add(Label::new(format!("Width: {width:#.2}")));
        ui.add(Label::new(format!("Height: {height:#.2}")));
        if let Some(walls) = &self.walls {
            ui.label("Walls:");
            for wall in walls {
                ui.add(Label::new(format!("{:?}", wall)));
            }
        }
        ui.add(Label::new(format!("Timelimit: {timelimit:#.2}")));
        ui.add(Label::new(format!("Step radius: {step_radius:#.2}")));
        ui.add(Label::new(format!("Term radius: {term_radius:#.2}")));
        if let Some(max_radius) = self.max_radius {
            ui.add(Label::new(format!("Max radius: {max_radius:#.2}")));
        }
        ui.add(Label::new(format!("Bounce factor: {bounce_factor:#.2}")));
        ui.add(Label::new(format!("Reward: {reward:?}")));
        ui.add(Label::new(format!("Seed: {seed:#.2}")));
    }

    fn render_mutable(
        &mut self,
        ui: &mut Ui,
    ) {
        ui.label("PointEnv Options");
        ui.add(Slider::new(&mut self.width, 0.0..=100.0).step_by(0.1));
        ui.add(Slider::new(&mut self.height, 0.0..=100.0).step_by(0.1));
        ui.add(Slider::new(&mut self.timelimit, 0..=1000));
        ui.add(Slider::new(&mut self.step_radius, 0.0..=10.0).step_by(0.1));
        ui.add(Slider::new(&mut self.term_radius, 0.0..=10.0).step_by(0.1));
        ui.add(Slider::new(&mut self.bounce_factor, 0.0..=1.0).step_by(0.01));
        ui.add(Slider::new(&mut self.seed, 0..=1000));
    }
}