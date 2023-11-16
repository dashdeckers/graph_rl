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

/// Generate a valid pair of (start, goal). \
/// A pair is valid if the goal is not reachable from the start within a single step,
/// and there are no wall collisions (i.e. start or goal is contained within a wall).
fn generate_start_goal(
    width: usize,
    height: usize,
    step_radius: f64,
    walls: &[PointLine],
    rng: &mut dyn RngCore,
) -> (PointState, PointState) {
    loop {
        let state = PointState::sample(rng, &[0.0..=(width as f64), 0.0..=(height as f64)]);
        let goal = PointState::sample(rng, &[0.0..=(width as f64), 0.0..=(height as f64)]);

        let wall_contains_state = walls
            .iter()
            .any(|w| w.contains(&state) || w.contains(&goal));

        if !wall_contains_state && !reachable(&state, &goal, step_radius, walls) {
            break (state, goal);
        }
    }
}

/// The goal is reachable from state if they are within step_radius of each other and no wall is in the way
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

/// Compute the next_state after taking action in state, considering any possible collisions with walls.
fn compute_next_state(
    width: usize,
    height: usize,
    state: PointState,
    action: &PointAction,
    step_radius: f64,
    bounce_factor: f64,
    walls: &[PointLine],
) -> PointState {
    // bounds on incoming action
    let action = action.restrict(step_radius);

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
                .map(|(idx, point)| (idx, point.squared_distance_to(&state)))
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
    next_state.restrict(width as f64, height as f64)
}

pub struct PointEnv {
    config: PointEnvConfig,
    width: usize,
    height: usize,
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
        debug_assert!(config.step_radius > 0.0 && config.step_radius <= 1.0);
        debug_assert!(config.bounce_factor > 0.0 && config.bounce_factor <= 1.0);
        debug_assert!(config.bounce_factor <= (config.step_radius / 10.0));
        // assertion guards for minimum map-size compared to step_radius
        debug_assert!(
            config.step_radius < (4.0 * config.width as f64)
                && config.step_radius < (4.0 * config.height as f64)
        );

        // add walls for the borders around the map
        let mut walls = config.walls.clone().unwrap_or_default();
        walls.extend([
            PointLine::from(((0.0, 0.0), (config.width as f64, 0.0))),
            PointLine::from(((0.0, 0.0), (0.0, config.height as f64))),
            PointLine::from((
                (config.width as f64, 0.0),
                (config.width as f64, config.height as f64),
            )),
            PointLine::from((
                (0.0, config.height as f64),
                (config.width as f64, config.height as f64),
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

        Ok(Box::new(PointEnv {
            config: config.clone(),
            width: config.width,
            height: config.height,
            walls,

            state: start,
            start,
            goal,
            history: vec![start],

            timestep: 0,
            timelimit: config.timelimit,
            reset_count: 0,

            step_radius: config.step_radius,
            term_radius: config.term_radius,
            bounce_factor: config.bounce_factor,
            reward: config.reward,

            rng,
        }))
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
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

    fn new(config: Self::Config) -> Result<Box<Self>> {
        Self::new(config)
    }

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
            self.step_radius,
            self.bounce_factor,
            &self.walls,
        );

        self.history.push(self.state);

        let reward = self
            .reward
            .compute(&self.state, &self.goal, self.term_radius, &self.walls);
        let terminated = reachable(&self.state, &self.goal, self.term_radius, &self.walls);
        let truncated = !terminated && (self.timestep == self.timelimit);

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

    fn timelimit(&self) -> usize {
        self.timelimit
    }

    fn action_space(&self) -> Vec<usize> {
        vec![2]
    }

    fn action_domain(&self) -> Vec<RangeInclusive<f64>> {
        vec![0.0..=self.step_radius]
    }

    fn observation_space(&self) -> Vec<usize> {
        vec![2 + 2] // + 4 * self.walls.len()]
    }

    fn observation_domain(&self) -> Vec<RangeInclusive<f64>> {
        vec![0.0..=(self.width as f64), 0.0..=(self.height as f64)]
    }

    fn current_observation(&self) -> Self::Observation {
        PointObs::from((self.state, self.goal, self.walls.as_ref()))
    }

    fn value_range(&self) -> (f64, f64) {
        let (lo, hi) = self
            .reward
            .value_range(self.timelimit, self.width, self.height);

        // add 40% padding to upper bound
        let padding = (lo.abs() + hi.abs()) * 0.4;

        (lo, hi + padding)
    }

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
            [self.width() as f64, self.height() as f64],
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
