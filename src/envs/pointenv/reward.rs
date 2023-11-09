use super::state::PointState;
use super::point_env::reachable;
use super::line::PointLine;
use super::super::DistanceMeasure;

#[derive(Debug)]
pub enum PointReward {
    Euclidean,
    Distance,
    Sparse,
}
impl PointReward {
    pub fn compute(
        &self,
        state: &PointState,
        goal: &PointState,
        term_radius: f64,
        walls: &[PointLine],
    ) -> f64 {
        match self {
            PointReward::Euclidean => -PointState::distance(state, goal),
            PointReward::Distance => if reachable(state, goal, term_radius, walls) {10.0} else {-1.0},
            PointReward::Sparse => if reachable(state, goal, term_radius, walls) {1.0} else {0.0},
        }
    }

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
            PointReward::Euclidean => (-((width.powi(2) + height.powi(2)).sqrt()) * timelimit, 0.0),
            PointReward::Distance => (-1.0 * timelimit, 0.0),
            PointReward::Sparse => (0.0, 1.0),
        }
    }
}