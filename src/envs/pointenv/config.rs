use {
    super::{
        line::PointLine,
        reward::PointReward,
    },
    crate::configs::RenderableConfig,
    strum::{
        EnumIter,
        IntoEnumIterator,
    },
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
        ComboBox,
    },
    std::fmt::Display,
};

/// An enum representing the different wall configurations for the
/// [`PointEnv`](super::point_env::PointEnv) environment.
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, EnumIter)]
pub enum PointEnvWalls {
    None,
    OneLine,
    TwoLine,
    FourLine,
    Hooks,
}
impl Display for PointEnvWalls {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::OneLine => write!(f, "OneLine"),
            Self::TwoLine => write!(f, "TwoLine"),
            Self::FourLine => write!(f, "FourLine"),
            Self::Hooks => write!(f, "Hooks"),
        }
    }
}
impl PointEnvWalls {
    pub fn to_walls(
        &self,
        width: f64,
        height: f64,
    ) -> Vec<PointLine> {
        let mut walls = match self {
            Self::None => Vec::new(),
            Self::OneLine => vec![
                PointLine::from(((0.0, height * 0.5), (width * 0.8, height * 0.5))),
            ],
            Self::TwoLine => vec![
                PointLine::from(((0.0, height * 0.2), (width * 0.8, height * 0.2))),
                PointLine::from(((width * 0.2, height * 0.8), (width, height * 0.8))),
            ],
            Self::FourLine => vec![
                PointLine::from(((width * 0.2, height * 0.8), (width, height * 0.8))),
                PointLine::from(((0.0, height * 0.6), (width * 0.8, height * 0.6))),
                PointLine::from(((width * 0.2, height * 0.4), (width, height * 0.4))),
                PointLine::from(((0.0, height * 0.2), (width * 0.8, height * 0.2))),
            ],
            Self::Hooks => vec![
                PointLine::from(((0.0, height * 0.2), (width * 0.7, height * 0.2))),
                PointLine::from(((width * 0.7, height * 0.6), (width * 0.7, height * 0.2))),

                PointLine::from(((width * 0.3, height * 0.8), (width, height * 0.8))),
                PointLine::from(((width * 0.3, height * 0.8), (width * 0.3, height * 0.4))),
            ],
        };
        walls.extend([
            PointLine::from(((0.0, 0.0), (width, 0.0))),
            PointLine::from(((0.0, 0.0), (0.0, height))),
            PointLine::from((
                (width, 0.0),
                (width, height),
            )),
            PointLine::from((
                (0.0, height),
                (width, height),
            )),
        ]);

        walls
    }
}

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
///     PointEnvWalls,
///     PointReward,
/// };
///
/// let config = PointEnvConfig::default();
/// assert_eq!(config.width, 5);
/// assert_eq!(config.height, 5);
/// assert_eq!(config.walls, PointEnvWalls::None);
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
    pub walls: PointEnvWalls,
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
            walls: PointEnvWalls::None,
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
        walls: PointEnvWalls,
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
        let walls = &self.walls;
        let timelimit = self.timelimit;
        let step_radius = self.step_radius;
        let term_radius = self.term_radius;
        let bounce_factor = self.bounce_factor;
        let reward = &self.reward;
        let seed = self.seed;

        ui.separator();
        ui.label("PointEnv Options");
        ui.add(Label::new(format!("Width: {width:#.2}")));
        ui.add(Label::new(format!("Height: {height:#.2}")));
        ui.add(Label::new(format!("Walls: {walls:#}")));
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
        ui.separator();
        ui.label("PointEnv Options");
        ui.add(
            Slider::new(&mut self.width, 1.0..=30.0)
            .step_by(0.1)
            .text("Width"),
        );
        ui.add(
            Slider::new(&mut self.height, 1.0..=30.0)
            .step_by(0.1)
            .text("Height"),
        );
        ComboBox::from_label("Walls")
            .selected_text(format!("{}", self.walls))
            .show_ui(ui, |ui| {
                for wall in PointEnvWalls::iter() {
                    ui.selectable_value(
                        &mut self.walls,
                        wall.clone(),
                        format!("{}", wall),
                    );
                }
            }
        );
        ui.add(
            Slider::new(&mut self.timelimit, 1..=1000)
            .text("Timelimit")
        );
        ui.add(
            Slider::new(&mut self.step_radius, 0.1..=10.0)
            .step_by(0.1)
            .text("Step radius")
        );
        ui.add(
            Slider::new(&mut self.term_radius, 0.1..=10.0)
            .step_by(0.1)
            .text("Term radius")
        );
        let mut max_radius = self.max_radius.unwrap_or(0.0);
        ui.add(
            Slider::new(&mut max_radius, 0.0..=10.0)
            .step_by(0.1)
            .text("Max radius")
        );
        self.max_radius = if max_radius == 0.0 {
            None
        } else {
            Some(max_radius)
        };
        ui.add(
            Slider::new(&mut self.bounce_factor, 0.1..=1.0)
            .step_by(0.01)
            .text("Bounce factor")
        );
        ui.add(
            Slider::new(&mut self.seed, 0..=1000)
            .text("Seed")
        );
    }
}