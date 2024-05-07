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
    serde::{
        Serialize,
        Deserialize,
    },
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
    anyhow::Result,
};

/// An enum representing the different wall configurations for the
/// [`PointEnv`](super::point_env::PointEnv) environment.
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize, EnumIter)]
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
/// * `spawn_radius_max` - The maximum distance allowed between the randomly generated start and goal.
/// * `spawn_radius_max` - The maximum distance allowed between the randomly generated start and goal.
/// * `spawn_centers` - When these points are given, then spawn start and goal with set radius around those points.
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
/// assert_eq!(config.spawn_radius_max, None);
/// assert_eq!(config.bounce_factor, 0.1);
/// assert_eq!(config.reward, PointReward::Distance);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointEnvConfig {
    pub width: f64,
    pub height: f64,
    pub walls: PointEnvWalls,
    pub timelimit: usize,
    pub step_radius: f64,
    pub term_radius: f64,
    pub spawn_radius_max: Option<f64>,
    pub spawn_radius_min: Option<f64>,
    pub spawn_centers: Option<((f64, f64), (f64, f64))>,
    pub bounce_factor: f64,
    pub reward: PointReward,
    pub seed: u64,
}
impl Default for PointEnvConfig {
    fn default() -> Self {
        Self {
            width: 10.0,
            height: 10.0,
            walls: PointEnvWalls::None,
            timelimit: 100,
            step_radius: 1.0,
            term_radius: 0.5,
            spawn_radius_max: None,
            spawn_radius_min: None,
            spawn_centers: None,
            bounce_factor: 0.1,
            reward: PointReward::Sparse,
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
        spawn_radius_max: Option<f64>,
        spawn_radius_min: Option<f64>,
        spawn_centers: Option<((f64, f64), (f64, f64))>,
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
            spawn_radius_max,
            spawn_radius_min,
            spawn_centers,
            bounce_factor,
            reward,
            seed,
        }
    }

    pub fn check(&self) -> Result<()> {
        if !(self.step_radius > 0.0 && self.step_radius <= 1.0) {
            return Err(anyhow::anyhow!("Step radius must be in the range (0.0, 1.0]"));
        }

        if !(self.bounce_factor > 0.0 && self.bounce_factor <= 1.0) {
            return Err(anyhow::anyhow!("Bounce factor must be in the range (0.0, 1.0]"));
        }

        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(self.bounce_factor <= (self.step_radius / 10.0)) {
            return Err(anyhow::anyhow!("Bounce factor must be less than or equal to step radius / 10.0"));
        }

        if !((self.step_radius * 4.0) < self.width && (self.step_radius * 4.0) < self.height) {
            return Err(anyhow::anyhow!("Step radius * 4.0 must be less than width and height"));
        }

        if self.spawn_centers.is_none() {
            if let Some(spawn_radius) = self.spawn_radius_max {
                if spawn_radius <= self.step_radius {
                    return Err(anyhow::anyhow!("Spawn radius max must be greater than or equal to step radius if spawn centers are not set"));
                }
            }
        }

        Ok(())
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
        let spawn_radius_max = self.spawn_radius_max;
        let spawn_radius_min = self.spawn_radius_min;
        let spawn_centers = self.spawn_centers;
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
        if let Some(spawn_radius_max) = spawn_radius_max {
            ui.add(Label::new(format!("Max radius: {spawn_radius_max:#.2}")));
        }
        if let Some(spawn_radius_min) = spawn_radius_min {
            ui.add(Label::new(format!("Min radius: {spawn_radius_min:#.2}")));
        }
        ui.add(Label::new(format!("Spawn centers: {spawn_centers:?}")));
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
        let mut spawn_radius_max = self.spawn_radius_max.unwrap_or(0.0);
        ui.add(
            Slider::new(&mut spawn_radius_max, 0.0..=10.0)
            .step_by(0.1)
            .text("Max radius")
        );
        self.spawn_radius_max = if spawn_radius_max == 0.0 {
            None
        } else {
            Some(spawn_radius_max)
        };
        let mut spawn_radius_min = self.spawn_radius_min.unwrap_or(0.0);
        ui.add(
            Slider::new(&mut spawn_radius_min, 0.0..=10.0)
            .step_by(0.1)
            .text("Min radius")
        );
        self.spawn_radius_min = if spawn_radius_min == 0.0 {
            None
        } else {
            Some(spawn_radius_min)
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


        // Allow the user to either set spawn centers for start and goal or leave them to be randomly generated
        let mut use_spawn_centers = self.spawn_centers.is_some();
        ui.checkbox(&mut use_spawn_centers, "Use spawn centers");
        self.spawn_centers = if !use_spawn_centers {
            None
        } else {
            let mut spawn_centers = self.spawn_centers.unwrap_or(((0.0, 0.0), (0.0, 0.0)));
            ui.add(
                Slider::new(&mut spawn_centers.0 .0, 0.0..=self.width)
                .step_by(0.1)
                .text("Start x")
            );
            ui.add(
                Slider::new(&mut spawn_centers.0 .1, 0.0..=self.height)
                .step_by(0.1)
                .text("Start y")
            );
            ui.add(
                Slider::new(&mut spawn_centers.1 .0, 0.0..=self.width)
                .step_by(0.1)
                .text("Goal x")
            );
            ui.add(
                Slider::new(&mut spawn_centers.1 .1, 0.0..=self.height)
                .step_by(0.1)
                .text("Goal y")
            );
            Some(spawn_centers)
        };
    }
}