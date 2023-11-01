#![allow(non_snake_case)]

use core::convert::Into;
use core::hash::Hash;
use std::ops::Range;
use std::fmt::{Display, Debug};

use derive_getters::Getters;
use auto_ops::impl_op_ex;
use tracing::{instrument, info, warn};
use ordered_float::OrderedFloat;
use rand::{RngCore, Rng, SeedableRng, rngs::StdRng};
use pyo3::prelude::*;
// use anyhow::{anyhow, Error};
use anyhow::Result;
use candle_core::Tensor;

use super::{Environment, Step};


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PointAction {
    dx: OrderedFloat<f64>,
    dy: OrderedFloat<f64>,
}
impl PointAction {
    pub fn dx(&self) -> f64 {
        self.dx.into_inner()
    }

    pub fn dy(&self) -> f64 {
        self.dy.into_inner()
    }

    pub fn sample(rng: &mut dyn RngCore, bounds: [Range<f64>; 1]) -> Self {
        let r: f64 = bounds[0].end * f64::sqrt(rng.gen_range(0.0..=1.0));
        let theta: f64 = rng.gen_range(0.0..=1.0) * 2.0 * std::f64::consts::PI;

        Self::from((
            r * theta.cos(),
            r * theta.sin(),
        ))
    }

    pub fn restrict(self, bounds: [Range<f64>; 1]) -> Self {
        let zero = PointState::from((0.0, 0.0));
        let step = PointState::from((self.dx(), self.dy()));
        let radius = bounds[0].end;

        let step_distance = zero.squared_distance_to(&step);

        if OrderedFloat(step_distance) <= OrderedFloat(radius.powi(2)) {
            Self::from((
                self.dx(),
                self.dy(),
            ))
        } else {
            let ratio = radius / step_distance;
            Self::from((
                self.dx() * ratio,
                self.dy() * ratio,
            ))
        }
    }

    pub fn distance(&self, other: &Self) -> f64 {
        let p1 = PointState::from(Into::<(f64, f64)>::into(*self));
        let p2 = PointState::from(Into::<(f64, f64)>::into(*other));
        p1.distance(&p2)
    }
}
// Convert (f64, f64) into PointAction
impl From<(f64, f64)> for PointAction {
    fn from(value: (f64, f64)) -> Self {
        Self {
            dx: OrderedFloat(value.0),
            dy: OrderedFloat(value.1),
        }
    }
}
// Convert PointAction into (f64, f64)
impl From<PointAction> for (f64, f64) {
    fn from(val: PointAction) -> Self {
        (val.dx(), val.dy())
    }
}
// Convert Vec<f64> into PointAction
impl From<Vec<f64>> for PointAction {
    fn from(value: Vec<f64>) -> Self {
        assert!(value.len() == 2);
        Self {
            dx: OrderedFloat(value[0]),
            dy: OrderedFloat(value[1]),
        }
    }
}
// Convert Vec<f64> into PointAction
impl From<PointAction> for Vec<f64> {
    fn from(value: PointAction) -> Self {
        vec![value.dx(), value.dy()]
    }
}
// Convert Tensor into PointAction
impl From<Tensor> for PointAction {
    fn from(value: Tensor) -> Self {
        let values = value
            .squeeze(0).unwrap()
            .to_vec1::<f64>().unwrap();
        Self::from(values)
    }
}
// Convert PointAction into Tensor
impl From<PointAction> for Tensor {
    fn from(value: PointAction) -> Self {
        Self::new(&[*value.dx, *value.dy], &candle_core::Device::Cpu).unwrap()
    }
}
// Convert PointAction into PyAny
impl IntoPy<PyObject> for PointAction {
    fn into_py(self, py: Python<'_>) -> PyObject {
        (*self.dx, *self.dy).into_py(py)
    }
}
// Display
impl Display for PointAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A({:.2}, {:.2})", self.dx, self.dy)
    }
}
// PointState + PointAction AND reference types
impl_op_ex!(+ |p1: &PointState, a: &PointAction| -> PointState {
    PointState::from((
        p1.x() + a.dx(),
        p1.y() + a.dy(),
    ))
});





#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PointState {
    x: OrderedFloat<f64>,
    y: OrderedFloat<f64>,
}
impl PointState {
    pub fn x(&self) -> f64 {
        self.x.into_inner()
    }

    pub fn y(&self) -> f64 {
        self.y.into_inner()
    }

    pub fn squared_distance_to(&self, other: &Self) -> f64 {
        let dx = self.x() - other.x();
        let dy = self.y() - other.y();
        dx.powi(2) + dy.powi(2)
    }

    pub fn magnitude(&self) -> f64 {
        self.squared_distance_to(&(0.0, 0.0).into()).sqrt()
    }

    pub fn in_radius_of(
        &self,
        other: &Self,
        radius: f64,
    ) -> bool {
        OrderedFloat(self.squared_distance_to(other)) <= OrderedFloat(radius.powi(2))
    }

    pub fn sample(rng: &mut dyn RngCore, bounds: [Range<f64>; 2]) -> Self {
        Self::from((
            rng.gen_range(bounds[0].start..bounds[0].end),
            rng.gen_range(bounds[1].start..bounds[1].end),
        ))
    }

    pub fn restrict(self, bounds: [Range<f64>; 2]) -> Self {
        Self::from((
            self.x().clamp(bounds[0].start, bounds[0].end - f64::EPSILON),
            self.y().clamp(bounds[1].start, bounds[1].end - f64::EPSILON),
        ))
    }

    pub fn distance(&self, other: &Self) -> f64 {
        self.squared_distance_to(other).sqrt()
    }
}
// Convert (f64, f64) into PointState
impl From<(f64, f64)> for PointState {
    fn from(value: (f64, f64)) -> Self {
        Self {
            x: OrderedFloat(value.0),
            y: OrderedFloat(value.1),
        }
    }
}
// Convert PointState into (f64, f64)
impl From<PointState> for (f64, f64) {
    fn from(val: PointState) -> Self {
        (val.x(), val.y())
    }
}
// Convert Vec<f64> into PointState
impl From<Vec<f64>> for PointState {
    fn from(value: Vec<f64>) -> Self {
        assert!(value.len() == 2);
        Self {
            x: OrderedFloat(value[0]),
            y: OrderedFloat(value[1]),
        }
    }
}
// Convert Vec<f64> into PointState
impl From<PointState> for Vec<f64> {
    fn from(value: PointState) -> Self {
        vec![value.x(), value.y()]
    }
}
// Convert Tensor into PointState
impl From<Tensor> for PointState {
    fn from(value: Tensor) -> Self {
        let values = value
            .squeeze(0).unwrap()
            .to_vec1::<f64>().unwrap();
        Self::from(values)
    }
}
// Convert PointState into Tensor
impl From<PointState> for Tensor {
    fn from(value: PointState) -> Self {
        Self::new(&[*value.x, *value.y], &candle_core::Device::Cpu).unwrap()
    }
}
// Display
impl Display for PointState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "S({:.2}, {:.2})", self.x, self.y)
    }
}
// PointState + PointState AND reference types
impl_op_ex!(+ |p1: &PointState, p2: &PointState| -> PointState {
    PointState {
        x: p1.x + p2.x,
        y: p1.y + p2.y,
    }
});
// PointState - PointState AND reference types
impl_op_ex!(- |p1: &PointState, p2: &PointState| -> PointState {
    PointState {
        x: p1.x - p2.x,
        y: p1.y - p2.y,
    }
});
// PointState * f64 AND reference types
impl_op_ex!(* |p1: &PointState, s: &f64| -> PointState {
    PointState {
        x: p1.x * s,
        y: p1.y * s,
    }
});
// f64 * PointState AND reference types
impl_op_ex!(* |s: &f64, p1: &PointState| -> PointState {
    PointState {
        x: p1.x * s,
        y: p1.y * s,
    }
});
// PointState * OrderedFloat(f64) AND reference types
impl_op_ex!(* |p1: &PointState, s: &OrderedFloat<f64>| -> PointState {
    PointState {
        x: p1.x * s,
        y: p1.y * s,
    }
});
// OrderedFloat(f64) * PointState AND reference types
impl_op_ex!(* |s: &OrderedFloat<f64>, p1: &PointState| -> PointState {
    PointState {
        x: p1.x * s,
        y: p1.y * s,
    }
});
// PointState / f64 AND reference types
impl_op_ex!(/ |p1: &PointState, s: &f64| -> PointState {
    if OrderedFloat(*s) == OrderedFloat(0.0) {
        panic!("Division by zero is not allowed");
    }
    PointState {
        x: p1.x / s,
        y: p1.y / s,
    }
});
// PointState / OrderedFloat(f64) AND reference types
impl_op_ex!(/ |p1: &PointState, s: &OrderedFloat<f64>| -> PointState {
    if *s == OrderedFloat(0.0) {
        panic!("Division by zero is not allowed");
    }
    PointState {
        x: p1.x / s,
        y: p1.y / s,
    }
});




#[derive(Debug)]
pub struct PointLine {
    // Let's assume a directionality from A -> B where it makes sense
    pub A: PointState,
    pub B: PointState,
}
impl PointLine {
    pub fn contains(
        &self,
        P: PointState,
    ) -> bool {
        self.A.x().min(self.B.x()) <= P.x() && P.x() <= self.A.x().max(self.B.x())
        && self.A.y().min(self.B.y()) <= P.y() && P.y() <= self.A.y().max(self.B.y())
    }

    pub fn collision_with(
        &self,
        other: &Self,
    ) -> Option<PointState> {
        let start = self.A;
        let goal = self.B;
        let A = other.A;
        let B = other.B;

        let Gx_Sx = goal.x() - start.x();
        let Gy_Sy = goal.y() - start.y();
        let Bx_Ax = B.x() - A.x();
        let By_Ay = B.y() - A.y();

        let determinant = -Bx_Ax * Gy_Sy + Gx_Sx * By_Ay;

        // if the determinant is zero, we have parallel or collinear lines
        // parallel: have same slope, collinear: on same line
        if determinant.abs() < f64::EPSILON {
            match (self.contains(other.A), self.contains(other.B)) {
                (true, true) => {
                    if self.A.squared_distance_to(&other.A) < self.A.squared_distance_to(&other.B) {
                        Some(other.A)
                    } else {
                        Some(other.B)
                    }
                },
                (false, true) => Some(other.B),
                (true, false) => Some(other.A),
                (false, false) => None
            }
        } else {
            let s = OrderedFloat((
                 -Gy_Sy * (start.x() - A.x())
                + Gx_Sx * (start.y() - A.y())
            ) / determinant);

            let t = OrderedFloat((
                  Bx_Ax * (start.y() - A.y())
                - By_Ay * (start.x() - A.x())
            ) / determinant);

            let zero = OrderedFloat(0.0);
            let one = OrderedFloat(1.0);

            if s >= zero && s <= one && t >= zero && t <= one {
                Some(PointState::from((
                    start.x() + *(t * Gx_Sx),
                    start.y() + *(t * Gy_Sy),
                )))
            } else {
                None
            }
        }
    }

    /// Bounce back from obstacle in opposite direction, but no further than starting point
    #[instrument]
    pub fn bounce_from_obstacle(
        position: PointState,
        collision: PointState,
        bounce_factor: OrderedFloat<f64>,
    ) -> PointState {

        // the vector pointing in the opposite direction to the agents movement
        let back_vector = position - collision;

        // normalize it to a unit vector of length 1 and then multiply by bounce_factor
        let normalized_back_vector = back_vector / back_vector.magnitude();
        let bounce_back_vector = bounce_factor * normalized_back_vector;

        // determine the bounced_back state
        let new_state = collision + bounce_back_vector;

        info!(
            concat!(
                "\nBackvector: {back_vector:#?},",
                "\nNormalized: {normalized_back_vector:#?},",
                "\nScaled: {bounce_back_vector:#?}",
            ),
            back_vector = back_vector,
            normalized_back_vector = normalized_back_vector,
            bounce_back_vector = bounce_back_vector,
        );

        // if we would bounce back farther than the original state, we dont bounce
        if PointLine::from((new_state, collision)).contains(position) {
            position
        } else {
            new_state
        }
    }

}
// Convert ((f64, f64), (f64, f64)) into PointLine
impl From<((f64, f64), (f64, f64))> for PointLine {
    fn from(value: ((f64, f64), (f64, f64))) -> Self {
        Self {
            A: PointState::from(value.0),
            B: PointState::from(value.1),
        }
    }
}
// Convert (PointState, PointState) into PointLine
impl From<(PointState, PointState)> for PointLine {
    fn from(value: (PointState, PointState)) -> Self {
        Self {
            A: value.0,
            B: value.1,
        }
    }
}
// Display
impl Display for PointLine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} -> {})", self.A, self.B)
    }
}










#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointReward {
    Euclidean,
    Sparse,
    SparseTimePenalty,
}
impl PointReward {
    pub fn compute(
        &self,
        state: &PointState,
        goal: &PointState,
    ) -> f64 {
        match self {
            PointReward::Euclidean => -state.distance(goal),
            PointReward::Sparse => if state == goal {1.0} else {0.0},
            PointReward::SparseTimePenalty => if state == goal {1.0} else {-1.0},
        }
    }
}
// // Display
// impl Display for Reward {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "R({:.2})", self.r())
//     }
// }







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
}
impl PointEnv {

    fn generate_start_goal(
        width: usize,
        height: usize,
        step_radius: f64,
        walls: &[PointLine],
        seed: u64,
    ) -> (PointState, PointState) {
        let mut rng = StdRng::seed_from_u64(seed);

        loop {
            let start = PointState::sample(&mut rng, [0.0..(width as f64), 0.0..(height as f64)]);
            let goal = PointState::sample(&mut rng, [0.0..(width as f64), 0.0..(height as f64)]);

            let wall_contains_state = walls.iter().any(|w| w.contains(start) || w.contains(goal));

            // no wall collisions and the goal is not reachable within a single step? we have a valid pair!
            if !wall_contains_state && !PointEnv::reachable(start, goal, step_radius, walls) {
                break (start, goal);
            }
        }
    }

    fn reachable(
        start: PointState,
        goal: PointState,
        step_radius: f64,
        walls: &[PointLine],
    ) -> bool {
        // goal is reachable from start if they are within radius of each other and no wall is in the way
        return
            start.in_radius_of(&goal, step_radius)
            && !walls.iter().any(|w| w.collision_with(&PointLine::from((start, goal))).is_some());
    }

    #[instrument(skip(self))]
    fn compute_next_state(
        &self,
        action: &PointAction,
    ) -> PointState {
        // bounds on incoming action
        let action = action.restrict([0.0..self.step_radius]);

        // make a line from A to B, collect any collisions
        let movement_line = PointLine::from((self.state, self.state + action));
        let collisions: Vec<PointState> = self.walls
            .iter()
            .filter_map(|w| w.collision_with(&movement_line))
            .collect();

        info!(
            concat!(
                "\nThe movement line is given by: {movement_line:#?},",
                "\nThe collisions are {collisions:#?}",
            ),
            movement_line = movement_line,
            collisions = collisions,
        );

        // decide the next state
        let next_state = match collisions.len() {

            // no collisions? easy
            0 => self.state + action,

            // one collision, bounce back from collision
            1 => PointLine::bounce_from_obstacle(
                self.state,
                collisions[0],
                OrderedFloat(self.bounce_factor),
            ),

            // multiple collisions, bounce back from closest collision
            _ => {
                let argclosest = collisions
                    .iter()
                    .enumerate()
                    .map(|(idx, point)| (idx, point.squared_distance_to(&self.state)))
                    .min_by(|(_, first), (_, second)| first.total_cmp(second))
                    .expect("Collisions cannot be an empty vector, we covered that case")
                    .0;

                PointLine::bounce_from_obstacle(
                    self.state,
                    collisions[argclosest],
                    OrderedFloat(self.bounce_factor),
                )
            },
        };

        // bounds on outgoing state
        next_state.restrict([0.0..(self.width as f64), 0.0..(self.height as f64)])
    }


    pub fn override_start_goal(
        &mut self,
        start: PointState,
        goal: PointState,
    ) {
        self.start = start;
        self.goal = goal;
    }
}


impl Environment for PointEnv {
    type Config = PointEnvConfig;
    type Action = PointAction;
    type State = PointState;

    #[instrument]
    #[allow(clippy::too_many_arguments)]
    fn new(
        config: Self::Config,
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
        let (start, goal) = PointEnv::generate_start_goal(
            config.width,
            config.height,
            config.step_radius,
            &walls,
            config.seed,
        );

        warn!(
            "New PointEnv ({width}, {height}) with start {start:#?} and goal {goal:#?}",
            width = config.width,
            height = config.height,
            start = start,
            goal = goal,
        );

        Ok(Box::new(PointEnv {
            width: config.width,
            height: config.height,
            walls,

            state: start,
            start,
            goal,
            history: Vec::new(),

            timestep: 0,
            timelimit: config.timelimit,
            reset_count: 0,

            step_radius: 1.0,
            bounce_factor: 0.1,
            reward: config.reward,
        }))
    }

    fn reset(
        &mut self,
        seed: u64,
    ) -> Result<Self::State> {
        self.timestep = 0;
        self.reset_count += 1;
        (self.start, self.goal) = PointEnv::generate_start_goal(
            self.width,
            self.height,
            self.step_radius,
            &self.walls,
            seed,
        );
        self.state = self.start;
        self.history = vec![self.state];

        warn!(
            "New PointEnv ({width}, {height}) with start {start:#?} and goal {goal:#?}",
            width = self.width,
            height = self.height,
            start = self.start,
            goal = self.goal,
        );

        Ok(self.state)
    }

    #[instrument(skip(self))]
    fn step(
        &mut self,
        action: PointAction,
    ) -> Result<Step<PointState, PointAction>> {
        self.timestep += 1;

        self.state = self.compute_next_state(&action);
        self.history.push(self.state);

        let reward = self.reward.compute(&self.state, &self.goal);
        let terminated = PointEnv::reachable(self.state, self.goal, self.step_radius, &self.walls);
        let truncated = !terminated && (self.timestep == self.timelimit);

        info!(
            concat!(
                "\nStepped with action {action:#?} to environment state {new_state:#?}",
                "\nwith reward {reward:#?} (terminated {terminated:#?}, truncated {truncated:#?})",
            ),
            action = action,
            new_state = self.state,
            reward = reward,
            terminated = terminated,
            truncated = truncated,
        );

        Ok(Step {
            state: self.state,
            action,
            reward,
            terminated,
            truncated,
        })
    }

    /// Returns the number of allowed actions for this environment.
    fn action_space(&self) -> usize {
        2
    }

    /// Returns the shape of the observation tensors.
    fn observation_space(&self) -> &[usize] {
        &[2]
    }

    fn current_state(&self) -> Self::State {
        self.state
    }

    fn current_goal(&self) -> Self::State {
        self.goal
    }
}

// impl Default for PointEnv {
//     fn default() -> Self {
//         let mut env = PointEnv::new(
//             5,
//             5,
//             None,
//             10,
//             1.0,
//             0.1,
//             PointReward::Euclidean,
//             42,
//         );
//         env.override_start_goal(
//             PointState::from((1.0, 1.0)),
//             PointState::from((4.0, 4.0)),
//         );
//         env
//     }
// }





// use crate::gym_env::Environment;

#[derive(Debug)]
pub struct PointEnvConfig {
    width: usize,
    height: usize,
    walls: Option<Vec<PointLine>>,
    timelimit: usize,
    step_radius: f64,
    bounce_factor: f64,
    reward: PointReward,
    seed: u64,
}
impl Default for PointEnvConfig {
    fn default() -> Self {
        Self {
            width: 5,
            height: 5,
            walls: None,
            timelimit: 30,
            step_radius: 1.0,
            bounce_factor: 0.1,
            reward: PointReward::Euclidean,
            seed: StdRng::from_entropy().gen::<u64>(),
        }
    }
}
impl PointEnvConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: usize,
        height: usize,
        walls: Option<Vec<PointLine>>,
        timelimit: usize,
        step_radius: f64,
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
            bounce_factor,
            reward,
            seed,
        }
    }
}















// #[cfg(test)]
// mod tests {
//     use super::*;

//     /// Tests a case that went wrong on a random-walk, exposing a bug in the
//     /// bounce_from_obstacle function where I subtracted points in the wrong
//     /// order to create the back_vector
//     #[test]
//     fn test_bounce_back_problemcase1() {
//         let mut env = PointEnv::new(
//             10,
//             10,
//             Some(vec![
//                 PointLine::from(((0.0, 5.0), (5.0, 5.0)))
//             ]),
//             PointState::from((8.00, 8.00)),
//             PointState::from((0.05, 0.98)),
//             100,

//             1.0,
//             0.1,
//             PointReward::Sparse,

//         );
//         let action = PointAction::from((-0.69, 0.63));
//         let rng = &mut rand::thread_rng();
//         env.step(&action, rng);

//         assert_eq!(env.state, PointState::from((0.05, 0.98)));
//     }

//     /// Tests a case where the agent was able to walk through a wall
//     /// State: (0.10, 1.92), Action: -> (-0.28, 0.68), Next State: (0.04, 2.07)
//     #[test]
//     fn test_walks_through_walls_problemcase1() {
//         let mut env = PointEnv::new(
//             5,
//             5,
//             Some(vec![
//                 PointLine::from(((0.0, 2.0), (2.0, 2.0))),
//                 PointLine::from(((2.0, 2.0), (2.0, 1.0))),
//                 PointLine::from(((2.0, 1.0), (3.0, 1.0))),
//             ]),
//             PointState::from((8.0, 8.0)),
//             PointState::from((0.103, 1.919)),
//             30,

//             1.0,
//             0.1,
//             PointReward::Sparse,

//         );
//         let action = PointAction::from((-0.28, 0.68));
//         let rng = &mut rand::thread_rng();
//         env.step(&action, rng);

//         assert!(env.state.y() < OrderedFloat(2.0));
//     }
// }



// #[cfg(test)]
// mod tests {
//     use super::*;

//     fn create_points() -> (PointState, PointState, PointState, PointState, PointState, PointState) {
//         let A = PointState::from((1.0, 1.0));
//         let B = PointState::from((4.0, 4.0));
//         let C = PointState::from((1.0, 4.0));
//         let D = PointState::from((4.0, 1.0));
//         let E = PointState::from((3.0, 3.0));
//         let F = PointState::from((2.5, 2.5));

//         (A, B, C, D, E, F)
//     }

//     #[test]
//     fn test_collision_1() {
//         let (A, B, C, D, _, _) = create_points();

//         let line1 = PointLine::from((A, B));
//         let line2 = PointLine::from((C, D));

//         assert_eq!(line1.collision_with(&line2), Some(PointState::from((2.5, 2.5))));
//     }

//     #[test]
//     fn test_collision_2() {
//         let (_, B, C, D, E, _) = create_points();

//         let line1 = PointLine::from((E, B));
//         let line2 = PointLine::from((C, D));

//         assert_eq!(line1.collision_with(&line2), None);
//     }

//     #[test]
//     fn test_collision_3() {
//         let (_, B, C, D, _, F) = create_points();

//         let line1 = PointLine::from((F, B));
//         let line2 = PointLine::from((C, D));

//         assert_eq!(line1.collision_with(&line2), Some(PointState::from((2.5, 2.5))));
//     }

//     #[test]
//     fn test_collision_4() {
//         let (A, B, _, _, E, F) = create_points();

//         let line1 = PointLine::from((A, F));
//         let line2 = PointLine::from((E, B));

//         assert_eq!(line1.collision_with(&line2), None);
//     }

//     #[test]
//     fn test_collision_5() {
//         let (A, B, _, _, E, F) = create_points();

//         let line1 = PointLine::from((A, E));
//         let line2 = PointLine::from((F, B));

//         assert_eq!(line1.collision_with(&line2), Some(PointState::from((2.5, 2.5))));
//     }

//     #[test]
//     fn test_collision_6() {
//         let (A, B, _, _, _, F) = create_points();

//         let line1 = PointLine::from((A, F));
//         let line2 = PointLine::from((F, B));

//         assert_eq!(line1.collision_with(&line2), Some(PointState::from((2.5, 2.5))));
//     }

//     #[test]
//     fn test_collision_7() {
//         let line1 = PointLine::from(((0.0, 0.0), (2.0, 2.0)));
//         let line2 = PointLine::from(((2.0, 2.0), (3.0, 3.0)));

//         assert_eq!(line1.collision_with(&line2), Some(PointState::from((2.0, 2.0))));
//     }

//     #[test]
//     fn test_collision_8() {
//         let line1 = PointLine::from(((0.0, 0.0), (2.0, 2.0)));
//         let line2 = PointLine::from(((1.0, 1.0), (3.0, 3.0)));

//         assert_eq!(line1.collision_with(&line2), Some(PointState::from((1.0, 1.0))));
//     }

// }