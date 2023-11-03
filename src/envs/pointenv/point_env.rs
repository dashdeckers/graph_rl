use ordered_float::OrderedFloat;
use derive_getters::Getters;
use rand::{SeedableRng, RngCore, rngs::StdRng};
use anyhow::Result;
use tracing::warn;

use super::state::PointState;
use super::action::PointAction;
use super::observation::PointObs;
use super::line::PointLine;
use super::reward::PointReward;
use super::config::PointEnvConfig;

use super::super::{Environment, Step};


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
        let state = PointState::sample(rng, width as f64, height as f64);
        let goal = PointState::sample(rng, width as f64, height as f64);

        let wall_contains_state = walls.iter().any(|w| w.contains(&state) || w.contains(&goal));

        if !wall_contains_state && !reachable(&state, &goal, step_radius, walls) {
            break (state, goal);
        }
    }
}

/// The goal is reachable from state if they are within step_radius of each other and no wall is in the way
fn reachable(
    state: &PointState,
    goal: &PointState,
    step_radius: f64,
    walls: &[PointLine],
) -> bool {
    return
        state.in_radius_of(goal, step_radius)
        && !walls.iter().any(|w| w.collision_with(&PointLine::from((*state, *goal))).is_some());
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
        1 => PointLine::bounce_from_obstacle(
            state,
            collisions[0],
            OrderedFloat(bounce_factor),
        ),

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
        },
    };

    warn!(
        concat!(
            "\nCompute next step:",
            "\nS({:.3}, {:.3}) + A({:.3}, {:.3}) --> S'({:.3}, {:.3})",
        ),
        state.x(), state.y(),
        action.dx(), action.dy(),
        next_state.x(), next_state.y(),
    );

    // bounds on outgoing state
    next_state.restrict(width as f64, height as f64)
}



#[derive(Getters)]
pub struct PointEnv {
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
    bounce_factor: f64,
    reward: PointReward,

    rng: StdRng,
}
impl PointEnv {
    fn new(
        config: PointEnvConfig,
    ) -> Result<Box<Self>> {
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
        let mut walls = config.walls.unwrap_or_default();
        walls.extend([
            PointLine::from(((0.0, 0.0), (config.width as f64, 0.0))),
            PointLine::from(((0.0, 0.0), (0.0, config.height as f64))),
            PointLine::from(((config.width as f64, 0.0), (config.width as f64, config.height as f64))),
            PointLine::from(((0.0, config.height as f64), (config.width as f64, config.height as f64))),
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

            step_radius: 1.0,
            bounce_factor: 0.1,
            reward: config.reward,

            rng,
        }))
    }
}


impl Environment for PointEnv {
    type Config = PointEnvConfig;
    type Action = PointAction;
    type Observation = PointObs;

    fn new(config: Self::Config) -> Result<Box<Self>> {
        Self::new(config)
    }

    fn reset(&mut self, seed: u64) -> Result<Self::Observation> {
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

    fn step(&mut self, action: Self::Action) -> Result<Step<Self::Observation, Self::Action>> {
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

        let reward = self.reward.compute(&self.state, &self.goal);
        let terminated = reachable(&self.state, &self.goal, self.step_radius, &self.walls);
        let truncated = !terminated && (self.timestep == self.timelimit);

        warn!(
            concat!(
                "\nPointEnv Step:",
                "\nS({:.3}, {:.3}) + G({:.3}, {:.3})",
                "\nA({:.3}, {:.3})",
                "\nR: {:?}",
            ),
            self.state.x(), self.state.y(), self.goal.x(), self.goal.y(),
            action.dx(), action.dy(),
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

    fn action_space(&self) -> Vec<usize> {
        vec![2]
    }

    fn observation_space(&self) -> Vec<usize> {
        vec![2 + 2 + 4 * self.walls.len()]
    }

    fn current_observation(&self) -> Self::Observation {
        PointObs::from((self.state, self.goal, self.walls.as_ref()))
    }
}
