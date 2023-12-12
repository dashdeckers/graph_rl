use {
    super::{
        super::{
            Environment,
            Renderable,
            Sampleable,
            Step,
        },
        action::PointAction,
        config::PointEnvConfig,
        line::PointLine,
        observation::PointObs,
        reward::PointReward,
        state::PointState,
    },
    anyhow::Result,
    egui::Color32,
    egui_plot::{
        Line,
        PlotBounds,
        PlotUi,
        Points,
    },
    ordered_float::OrderedFloat,
    rand::{
        rngs::StdRng,
        RngCore,
        SeedableRng,
    },
    std::ops::RangeInclusive,
    tracing::info,
};

/// Generate a valid `(start, goal)` pair of [PointState]s.
///
/// A pair is valid if the goal is not reachable from the start within a single
/// step, and there are no wall collisions i.e. neither start nor goal is
/// contained within a wall / [PointLine].
fn generate_start_goal(
    width: f64,
    height: f64,
    step_radius: f64,
    walls: &[PointLine],
    rng: &mut dyn RngCore,
) -> (PointState, PointState) {
    loop {
        let state = PointState::sample(rng, &[0.0..=width, 0.0..=height]);
        let goal = PointState::sample(rng, &[0.0..=width, 0.0..=height]);

        let wall_contains_state = walls
            .iter()
            .any(|w| w.contains(&state) || w.contains(&goal));

        if !wall_contains_state && !reachable(&state, &goal, step_radius, walls) {
            break (state, goal);
        }
    }
}

/// The goal is reachable from state if they are within `step_radius` of each
/// other and none of the `walls` would block a straight line between them.
pub fn reachable(
    state: &PointState,
    goal: &PointState,
    step_radius: f64,
    walls: &[PointLine],
) -> bool {
    return state.in_radius_of(goal, step_radius)
        && !walls.iter().any(|w| {
            w.collision_with(&PointLine::from((*state, *goal)))
                .is_some()
        });
}

/// Compute the `next_state` after taking `action` in `state`, considering any
/// possible collisions with `walls`.
fn compute_next_state(
    width: f64,
    height: f64,
    state: PointState,
    action: &PointAction,
    bounce_factor: f64,
    walls: &[PointLine],
) -> PointState {
    // make a line from A to B, collect any collisions
    let movement_line = PointLine::from((state, state + action));
    let collisions: Vec<PointState> = walls
        .iter()
        .filter_map(|w| w.collision_with(&movement_line))
        .collect();

    // decide the next state
    let next_state = match collisions.len() {
        // no collisions? easy
        0 => state + action,

        // one collision? bounce back from collision
        1 => PointLine::bounce_from_obstacle(state, collisions[0], OrderedFloat(bounce_factor)),

        // multiple collisions? bounce back from closest collision
        _ => {
            let argclosest = collisions
                .iter()
                .enumerate()
                .map(|(idx, point)| (idx, point.distance_to(&state)))
                .min_by(|(_, first), (_, second)| first.total_cmp(second))
                .expect("Collisions cannot be an empty vector, we covered that case")
                .0;

            PointLine::bounce_from_obstacle(
                state,
                collisions[argclosest],
                OrderedFloat(bounce_factor),
            )
        }
    };

    info!(
        concat!(
            "\nCompute next step:",
            "\nS({:.3}, {:.3}) + A({:.3}, {:.3}) --> S'({:.3}, {:.3})",
        ),
        state.x(),
        state.y(),
        action.dx(),
        action.dy(),
        next_state.x(),
        next_state.y(),
    );

    // bounds on outgoing state
    next_state.restrict(width, height)
}

/// A [PointEnv] is a 2D continuous action environment where the agent is a
/// point in space, and the goal is to reach another point in space.
///
/// The agent can move in any direction, but only within a radius given by
/// step size. The environment is bounded by walls, and more walls can be added
/// to the environment to make it more difficult.
///
/// If the agent collides with a wall, it will bounce off the wall by a factor
/// of the travelled distance. The angle of incidence here is equal to the angle
/// of reflection.
pub struct PointEnv {
    config: PointEnvConfig,
    width: f64,
    height: f64,
    walls: Vec<PointLine>,

    state: PointState,
    start: PointState,
    goal: PointState,
    history: Vec<PointState>,

    timestep: usize,
    timelimit: usize,
    reset_count: usize,

    step_radius: f64,
    term_radius: f64,
    bounce_factor: f64,
    reward: PointReward,

    rng: StdRng,
}
impl PointEnv {
    fn new(config: PointEnvConfig) -> Result<Box<Self>> {
        // assertion guards for valid parameter values
        assert!(config.step_radius > 0.0 && config.step_radius <= 1.0);
        assert!(config.bounce_factor > 0.0 && config.bounce_factor <= 1.0);
        assert!(config.bounce_factor <= (config.step_radius / 10.0));
        // assertion guards for minimum map-size compared to step_radius
        assert!(
            (config.step_radius * 4.0) < config.width
                && (config.step_radius * 4.0) < config.height
        );

        // add walls for the borders around the map
        let mut walls = config.walls.clone().unwrap_or_default();
        walls.extend([
            PointLine::from(((0.0, 0.0), (config.width, 0.0))),
            PointLine::from(((0.0, 0.0), (0.0, config.height))),
            PointLine::from((
                (config.width, 0.0),
                (config.width, config.height),
            )),
            PointLine::from((
                (0.0, config.height),
                (config.width, config.height),
            )),
        ]);

        // compute random start and goal
        let mut rng = StdRng::seed_from_u64(config.seed);
        let (start, goal) = generate_start_goal(
            config.width,
            config.height,
            config.step_radius,
            &walls,
            &mut rng,
        );


        // assertion guards for square map
        assert!(config.width == config.height);
        // scale the map down to a 1x1 square
        let scale = config.width;

        let width = config.width / scale;
        let height = config.height / scale;
        let walls = walls.into_iter().map(|l| l / scale).collect();
        let start = start / scale;
        let goal = goal / scale;
        let step_radius = config.step_radius / scale;
        let term_radius = config.term_radius / scale;
        let bounce_factor = config.bounce_factor / scale;

        // let width = config.width;
        // let height = config.height;
        // let step_radius = config.step_radius;
        // let term_radius = config.term_radius;
        // let bounce_factor = config.bounce_factor;


        Ok(Box::new(PointEnv {
            config: config.clone(),
            width,
            height,
            walls,

            state: start,
            start,
            goal,
            history: vec![start],

            timestep: 0,
            timelimit: config.timelimit,
            reset_count: 0,

            step_radius,
            term_radius,
            bounce_factor,
            reward: config.reward,

            rng,
        }))
    }

    pub fn width(&self) -> f64 {
        self.width
    }

    pub fn height(&self) -> f64 {
        self.height
    }

    pub fn walls(&self) -> &Vec<PointLine> {
        &self.walls
    }

    pub fn state(&self) -> &PointState {
        &self.state
    }

    pub fn start(&self) -> &PointState {
        &self.start
    }

    pub fn goal(&self) -> &PointState {
        &self.goal
    }

    pub fn history(&self) -> &Vec<PointState> {
        &self.history
    }
}

impl Environment for PointEnv {
    type Config = PointEnvConfig;
    type Action = PointAction;
    type Observation = PointObs;

    /// Create a new [PointEnv] with the given [PointEnvConfig].
    ///
    /// # Panics
    ///
    /// Panics if any of the following conditions are not met:
    /// - `config.step_radius` is in `(0.0, 1.0]`
    /// - `config.bounce_factor` is in `(0.0, 1.0]`
    /// - `config.bounce_factor` is less than `config.step_radius / 10.0`
    /// - `config.step_radius * 4.0` is less than `config.width`
    /// - `config.step_radius * 4.0` is less than `config.height`
    ///
    /// # Errors
    ///
    /// Returns an error if the `config.walls` are not valid.
    fn new(config: Self::Config) -> Result<Box<Self>> {
        Self::new(config)
    }

    /// Reset the environment to a new episode, with a new random start and goal.
    fn reset(
        &mut self,
        seed: u64,
    ) -> Result<Self::Observation> {
        self.timestep = 0;
        self.reset_count += 1;

        self.rng = StdRng::seed_from_u64(seed);
        (self.start, self.goal) = generate_start_goal(
            self.width,
            self.height,
            self.step_radius,
            &self.walls,
            &mut self.rng,
        );
        self.state = self.start;

        self.history = vec![self.start];

        Ok(PointObs::from((self.start, self.goal, self.walls.as_ref())))
    }

    /// Take a step in the environment, returning the new observation and reward.
    ///
    /// The return type is a [Step] struct, which contains the following fields:
    /// - `observation`: the new observation after taking the step
    /// - `action`: the action that was taken
    /// - `reward`: the reward for taking the action
    /// - `terminated`: whether the episode is terminated
    /// - `truncated`: whether the episode is truncated
    fn step(
        &mut self,
        action: Self::Action,
    ) -> Result<Step<Self::Observation, Self::Action>> {
        let action = action.restrict(self.step_radius);
        self.timestep += 1;

        self.state = compute_next_state(
            self.width,
            self.height,
            self.state,
            &action,
            self.bounce_factor,
            &self.walls,
        );

        self.history.push(self.state);

        let reward = self
            .reward
            .compute(&self.state, &self.goal, self.term_radius, &self.walls);
        let terminated = reachable(&self.state, &self.goal, self.term_radius, &self.walls);
        let truncated = !terminated && (self.timestep >= self.timelimit);

        info!(
            concat!(
                "\nPointEnv Step:",
                "\nS({:.3}, {:.3}) + G({:.3}, {:.3})",
                "\nA({:.3}, {:.3})",
                "\nR: {:?}",
            ),
            self.state.x(),
            self.state.y(),
            self.goal.x(),
            self.goal.y(),
            action.dx(),
            action.dy(),
            reward,
        );

        Ok(Step {
            observation: PointObs::from((self.state, self.goal, self.walls.as_ref())),
            action,
            reward,
            terminated,
            truncated,
        })
    }

    /// Return the maximum number of steps allowed before the episode is truncated.
    fn timelimit(&self) -> usize {
        self.timelimit
    }

    /// The action space of [PointEnv] is `[2]` (2D continuous actions).
    fn action_space(&self) -> Vec<usize> {
        vec![2]
    }

    /// The action domain of [PointEnv] is `[0.0..=step_radius]`
    fn action_domain(&self) -> Vec<RangeInclusive<f64>> {
        vec![0.0..=self.step_radius]
    }

    /// The observation space of [PointEnv] is `[2 + 2 + 4 * n]`, where `n` is
    /// the number of walls in the environment.
    ///
    /// - `2`: the x and y coordinates of the agent
    /// - `2`: the x and y coordinates of the goal
    /// - `4 * n`: the x and y coordinates of the start and end points of each wall
    fn observation_space(&self) -> Vec<usize> {
        vec![2 + 2] // + 4 * self.walls.len()]
    }

    /// The observation domain of [PointEnv] is `[0.0..=width, 0.0..=height; 4 + 4 * n]`,
    fn observation_domain(&self) -> Vec<RangeInclusive<f64>> {
        vec![0.0..=self.width, 0.0..=self.height]
    }

    /// Return the current observation of the environment.
    fn current_observation(&self) -> Self::Observation {
        PointObs::from((self.state, self.goal, self.walls.as_ref()))
    }

    /// Return the value range of the reward function, with a 40% padding on the upper bound.
    fn value_range(&self) -> (f64, f64) {
        let (lo, hi) = self
            .reward
            .value_range(self.timelimit, self.width, self.height);

        // add 40% padding to upper bound
        let padding = (lo.abs() + hi.abs()) * 0.4;

        (lo, hi + padding)
    }

    /// Return the [PointEnvConfig] used to create this environment.
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl Renderable for PointEnv {
    fn render(
        &mut self,
        plot_ui: &mut PlotUi,
    ) {
        // Setup plot bounds
        plot_ui.set_plot_bounds(PlotBounds::from_min_max(
            [0.0, 0.0],
            [self.width(), self.height()],
        ));
        // Plot walls
        for wall in self.walls().iter() {
            plot_ui.line(
                Line::new(vec![[wall.A.x(), wall.A.y()], [wall.B.x(), wall.B.y()]])
                    .width(2.0)
                    .color(Color32::WHITE),
            )
        }
        // Plot start and goal
        let start = self.start();
        plot_ui.points(
            Points::new(vec![[start.x(), start.y()]])
                .radius(2.0)
                .color(Color32::WHITE),
        );
        let goal = self.goal();
        plot_ui.points(
            Points::new(vec![[goal.x(), goal.y()]])
                .radius(2.0)
                .color(Color32::GREEN),
        );
        // Plot path
        plot_ui.line(Line::new(
            self.history()
                .iter()
                .map(|p| [p.x(), p.y()])
                .collect::<Vec<_>>(),
        ))
    }
}
