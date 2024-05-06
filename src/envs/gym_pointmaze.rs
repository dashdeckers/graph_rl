use {
    super::{
        gym_wrappers::{
            gym_create_env,
            gym_reset_env,
            gym_step_env,
        },
        DistanceMeasure,
        Environment,
        RenderableEnvironment,
        Sampleable,
        Step,
        TensorConvertible,
        VectorConvertible,
        GoalAwareObservation,
    },
    crate::configs::RenderableConfig,
    serde::{
        Serialize,
        Deserialize,
    },
    anyhow::Result,
    candle_core::{
        Device,
        Tensor,
    },
    egui::Color32,
    egui_plot::{
        Line,
        Points,
        PlotBounds,
        PlotUi,
    },
    ordered_float::OrderedFloat,
    pyo3::PyObject,
    rand::Rng,
    std::ops::RangeInclusive,
};

fn preprocess_action(mut value: Vec<f64>) -> Vec<f64> {
    value[0] = value[0].clamp(-1.0, 1.0);
    value[1] = value[1].clamp(-1.0, 1.0);
    value
}

#[allow(unused_mut)]
fn preprocess_state(mut value: Vec<f64>) -> Vec<f64> {
    value
}

#[allow(unused_mut)]
fn preprocess_view(mut value: Vec<f64>) -> Vec<f64> {
    value
}

#[derive(Clone)]
pub struct PointMazeEnv {
    config: PointMazeConfig,
    env: PyObject,
    #[allow(dead_code)]
    maze: Vec<Vec<char>>,
    width: usize,
    height: usize,
    timelimit: usize,
    reward_mode: PointMazeReward,
    current_observation: PointMazeObservation,
    action_space: Vec<usize>,
    observation_space: Vec<usize>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PointMazeConfig {
    name: String,
    maze: Vec<Vec<char>>,
    width: usize,
    height: usize,
    timelimit: usize,
    reward_mode: PointMazeReward,
    seed: u64,
}
impl Default for PointMazeConfig {
    fn default() -> Self {
        // TODO: this hardcodes the PointMaze_Open-v3
        // adapt gym_create_env to take some kwargs
        // and parse the env string with its options
        let name = "PointMaze_OpenDense-v3".to_owned();
        let maze = vec![
            vec!['1', '1', '1', '1', '1', '1', '1'],
            vec!['1', '0', '0', '0', '0', '0', '1'],
            vec!['1', '0', '0', '0', '0', '0', '1'],
            vec!['1', '0', '0', '0', '0', '0', '1'],
            vec!['1', '1', '1', '1', '1', '1', '1'],
        ];
        let width = maze[0].len();
        let height = maze.len();

        assert!(width < 12 && height < 9);
        Self {
            name,
            maze,
            width,
            height,
            timelimit: 300,
            reward_mode: PointMazeReward::Dense,
            seed: rand::thread_rng().gen::<u64>(),
        }
    }
}
impl RenderableConfig for PointMazeConfig {
    fn render_immutable(
        &self,
        ui: &mut egui::Ui,
    ) {
        ui.label("PointMazeEnv");
        ui.label(format!("name: {}", self.name));
        ui.label(format!("width: {}", self.width));
        ui.label(format!("height: {}", self.height));
        ui.label(format!("timelimit: {}", self.timelimit));
        ui.label(format!("reward_mode: {:?}", self.reward_mode));
        ui.label(format!("seed: {}", self.seed));
    }

    fn render_mutable(
            &mut self,
            ui: &mut egui::Ui,
        ) {
        self.render_immutable(ui);
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PointMazeReward {
    SparseNegative,
    Sparse,
    Dense,
}
impl PointMazeReward {
    pub fn value_range(
        &self,
        timelimit: usize,
        width: usize,
        height: usize,
    ) -> (f64, f64) {
        let timelimit = timelimit as f64;
        let width = width as f64;
        let height = height as f64;
        match self {
            PointMazeReward::Dense => (-((width.powi(2) + height.powi(2)).sqrt()) * timelimit, 0.0),
            PointMazeReward::SparseNegative => (-1.0 * timelimit, 0.0),
            PointMazeReward::Sparse => (0.0, 1.0),
        }
    }

}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PointMazeAction {
    // Linear force applied to the ball in the x and y directions
    force_x: OrderedFloat<f64>,
    force_y: OrderedFloat<f64>,
}
impl Sampleable for PointMazeAction {
    fn sample(
        rng: &mut dyn rand::RngCore,
        domain: &[RangeInclusive<f64>],
    ) -> Self {
        assert!(domain.len() == 2);
        Self {
            force_x: OrderedFloat(rng.gen_range(domain[0].clone())),
            force_y: OrderedFloat(rng.gen_range(domain[1].clone())),
        }
    }
}
impl VectorConvertible for PointMazeAction {
    fn from_vec_pp(value: Vec<f64>) -> Self {
        Self::from_vec(preprocess_action(value))
    }
    fn from_vec(value: Vec<f64>) -> Self {
        // Make sure the number of elements in the Vec makes sense
        assert!(value.len() == 2);
        Self {
            // Preprocess the action
            force_x: OrderedFloat(value[0]),
            force_y: OrderedFloat(value[1]),
        }
    }
    fn to_vec(value: Self) -> Vec<f64> {
        vec![*value.force_x, *value.force_y]
    }
}
impl TensorConvertible for PointMazeAction {
    fn from_tensor_pp(value: Tensor) -> Self {
        Self::from_vec_pp(value.to_vec1::<f64>().unwrap())
    }
    fn from_tensor(value: Tensor) -> Self {
        Self::from_vec(value.to_vec1::<f64>().unwrap())
    }
    fn to_tensor(
        value: Self,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        Tensor::new(Self::to_vec(value), device)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PointMazeState {
    // The (x, y) coordinates of the ball
    x: OrderedFloat<f64>,
    y: OrderedFloat<f64>,
}
impl VectorConvertible for PointMazeState {
    fn from_vec_pp(value: Vec<f64>) -> Self {
        Self::from_vec(preprocess_state(value))
    }
    fn from_vec(value: Vec<f64>) -> Self {
        // Make sure the number of elements in the Vec makes sense
        assert!(value.len() == 2);
        Self {
            x: OrderedFloat(value[0]),
            y: OrderedFloat(value[1]),
        }
    }
    fn to_vec(value: Self) -> Vec<f64> {
        vec![*value.x, *value.y]
    }
}
impl TensorConvertible for PointMazeState {
    fn from_tensor_pp(value: Tensor) -> Self {
        Self::from_vec_pp(value.to_vec1::<f64>().unwrap())
    }
    fn from_tensor(value: Tensor) -> Self {
        Self::from_vec(value.to_vec1::<f64>().unwrap())
    }
    fn to_tensor(
        value: Self,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        Tensor::new(Self::to_vec(value), device)
    }
}
impl DistanceMeasure for PointMazeState {
    fn distance(
        s1: &Self,
        s2: &Self,
    ) -> f64 {
        ((s1.x - s2.x).powi(2) + (s1.y - s2.y).powi(2)).sqrt()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PointMazeView {
    // The (x, y) coordinates of the ball
    x: OrderedFloat<f64>,
    y: OrderedFloat<f64>,
    // The (x, y) velocities of the ball
    velocity_x: OrderedFloat<f64>,
    velocity_y: OrderedFloat<f64>,
}
impl VectorConvertible for PointMazeView {
    fn from_vec_pp(value: Vec<f64>) -> Self {
        Self::from_vec(preprocess_view(value))
    }
    fn from_vec(value: Vec<f64>) -> Self {
        // Make sure the number of elements in the Vec makes sense
        assert!(value.len() == 4);
        Self {
            x: OrderedFloat(value[0]),
            y: OrderedFloat(value[1]),
            velocity_x: OrderedFloat(value[2]),
            velocity_y: OrderedFloat(value[3]),
        }
    }
    fn to_vec(value: Self) -> Vec<f64> {
        vec![*value.x, *value.y, *value.velocity_x, *value.velocity_y]
    }
}
impl TensorConvertible for PointMazeView {
    fn from_tensor_pp(value: Tensor) -> Self {
        Self::from_vec_pp(value.to_vec1::<f64>().unwrap())
    }
    fn from_tensor(value: Tensor) -> Self {
        Self::from_vec(value.to_vec1::<f64>().unwrap())
    }
    fn to_tensor(
        value: Self,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        Tensor::new(Self::to_vec(value), device)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PointMazeObservation {
    observation: PointMazeView,
    desired_goal: PointMazeState,
    achieved_goal: PointMazeState,
}
impl GoalAwareObservation for PointMazeObservation {
    type State = PointMazeState;
    type View = PointMazeView;

    /// The achieved goal is the current [PointMazeState]
    fn achieved_goal(&self) -> &Self::State {
        &self.achieved_goal
    }

    /// The desired goal is the goal [PointMazeState]
    fn desired_goal(&self) -> &Self::State {
        &self.desired_goal
    }

    /// The observation is the current [PointMazeView]
    fn observation(&self) -> &Self::View {
        &self.observation
    }

    /// Set the achieved goal to the given value
    fn set_achieved_goal(
        &mut self,
        value: &Self::State,
    ) {
        self.achieved_goal = value.clone();
    }

    /// Set the desired goal to the given value
    fn set_desired_goal(
        &mut self,
        value: &Self::State,
    ) {
        self.desired_goal = value.clone();
    }

    /// Set the observation to the given value
    fn set_observation(
        &mut self,
        value: &Self::View,
    ) {
        self.observation = value.clone();
    }

    /// Create a new [PointMazeObservation] from the given values
    fn new(
        achieved_goal: &Self::State,
        desired_goal: &Self::State,
        observation: &Self::View,
    ) -> Self {
        Self {
            observation: observation.clone(),
            desired_goal: desired_goal.clone(),
            achieved_goal: achieved_goal.clone(),
        }
    }
}
impl VectorConvertible for PointMazeObservation {
    fn from_vec_pp(value: Vec<f64>) -> Self {
        // Make sure the number of elements in the Vec makes sense
        assert!(value.len() == 8);
        Self {
            observation: PointMazeView::from_vec_pp(value[0..4].to_vec()),
            desired_goal: PointMazeState::from_vec_pp(value[4..6].to_vec()),
            achieved_goal: PointMazeState::from_vec_pp(value[6..8].to_vec()),
        }
    }
    fn from_vec(value: Vec<f64>) -> Self {
        // Make sure the number of elements in the Vec makes sense
        assert!(value.len() == 8);
        Self {
            observation: PointMazeView::from_vec(value[0..4].to_vec()),
            desired_goal: PointMazeState::from_vec(value[4..6].to_vec()),
            achieved_goal: PointMazeState::from_vec(value[6..8].to_vec()),
        }
    }
    fn to_vec(value: Self) -> Vec<f64> {
        let mut vec = Vec::new();
        vec.extend(PointMazeView::to_vec(value.observation));
        vec.extend(PointMazeState::to_vec(value.desired_goal));
        vec.extend(PointMazeState::to_vec(value.achieved_goal));
        vec
    }
}
impl TensorConvertible for PointMazeObservation {
    fn from_tensor_pp(value: Tensor) -> Self {
        Self::from_vec_pp(value.to_vec1::<f64>().unwrap())
    }
    fn from_tensor(value: Tensor) -> Self {
        Self::from_vec(value.to_vec1::<f64>().unwrap())
    }
    fn to_tensor(
        value: Self,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        Tensor::new(Self::to_vec(value), device)
    }
}
impl DistanceMeasure for PointMazeObservation {
    fn distance(
        s1: &Self,
        s2: &Self,
    ) -> f64 {
        PointMazeState::distance(&s1.achieved_goal, &s2.achieved_goal)
    }
}

impl Environment for PointMazeEnv {
    type Config = PointMazeConfig;
    type Action = PointMazeAction;
    type Observation = PointMazeObservation;

    fn new(config: Self::Config) -> Result<Box<Self>> {
        let (env, action_space, observation_space) = gym_create_env(&config.name, true)?;
        let current_observation = gym_reset_env(&env, config.seed, true)?;
        Ok(Box::new(Self {
            config: config.clone(),
            env,
            maze: config.maze,
            width: config.width,
            height: config.height,
            timelimit: config.timelimit,
            reward_mode: config.reward_mode,
            current_observation,
            action_space,
            observation_space,
        }))
    }

    fn reset(
        &mut self,
        seed: u64,
    ) -> Result<Self::Observation> {
        self.current_observation = gym_reset_env(&self.env, seed, true)?;
        Ok(self.current_observation())
    }

    fn step(
        &mut self,
        action: Self::Action,
    ) -> Result<Step<Self::Observation, Self::Action>> {
        let mut step: Step<Self::Observation, Self::Action> = gym_step_env(&self.env, action, true)?;
        self.current_observation = step.observation.clone();
        if let PointMazeReward::SparseNegative = self.reward_mode {
            step.reward = -1.0;
        };
        Ok(step)
    }

    fn timelimit(&self) -> usize {
        self.timelimit
    }

    fn action_space(&self) -> Vec<usize> {
        self.action_space.clone()
    }

    fn action_domain(&self) -> Vec<RangeInclusive<f64>> {
        vec![-1.0..=1.0, -1.0..=1.0]
    }

    fn observation_space(&self) -> Vec<usize> {
        self.observation_space.clone()
    }

    fn observation_domain(&self) -> Vec<RangeInclusive<f64>> {
        vec![
            -f64::INFINITY..=f64::INFINITY,
            -f64::INFINITY..=f64::INFINITY,
            -f64::INFINITY..=f64::INFINITY,
            -f64::INFINITY..=f64::INFINITY,
            -f64::INFINITY..=f64::INFINITY,
            -f64::INFINITY..=f64::INFINITY,
            -f64::INFINITY..=f64::INFINITY,
            -f64::INFINITY..=f64::INFINITY,
        ]
    }

    fn current_observation(&self) -> Self::Observation {
        self.current_observation.clone()
    }

    fn value_range(&self) -> (f64, f64) {
        let (lo, hi) = self
            .reward_mode
            .value_range(self.timelimit, self.width, self.height);

        // add 40% padding to upper bound
        let padding = (lo.abs() + hi.abs()) * 0.4;

        (lo, hi + padding)
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl RenderableEnvironment for PointMazeEnv {
    fn render(
        &mut self,
        plot_ui: &mut PlotUi,
    ) {
        // Setup plot bounds
        let inner_width = (self.width - 2) as f64;
        let inner_height = (self.height - 2) as f64;
        let (min_x, max_x) = (-(inner_width / 2.0), (inner_width / 2.0));
        let (min_y, max_y) = (-(inner_height / 2.0), (inner_height / 2.0));
        plot_ui.set_plot_bounds(PlotBounds::from_min_max([min_x, min_y], [max_x, max_y]));
        // Draw the PointMaze
        let walls = vec![
            // Top wall
            (
                (min_x, max_y),
                (max_x, max_y),
            ),
            // Bottom wall
            (
                (min_x, min_y),
                (max_x, min_y),
            ),
            // Left wall
            (
                (min_x, min_y),
                (min_x, max_y),
            ),
            // Right wall
            (
                (max_x, min_y),
                (max_x, max_y),
            ),
        ];
        for ((x1, y1), (x2, y2)) in walls {
            plot_ui.line(
                Line::new(vec![[x1, y1], [x2, y2]])
                    .width(3.0)
                    .color(Color32::WHITE),
            )
        }
        let obs = self.current_observation();
        let state = &obs.achieved_goal;
        let goal = &obs.desired_goal;

        plot_ui.points(
            Points::new(vec![[*goal.x, *goal.y]])
                .radius(2.0)
                .color(Color32::GREEN),
        );
        plot_ui.points(
            Points::new(vec![[*state.x, *state.y]])
                .radius(2.0)
                .color(Color32::RED),
        );
    }
}
