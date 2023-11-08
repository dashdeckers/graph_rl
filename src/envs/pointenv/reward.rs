use super::state::PointState;
use super::super::DistanceMeasure;

#[derive(Debug)]
pub enum PointReward {
    Euclidean,
    Distance,
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
            PointReward::Euclidean => -PointState::distance(state, goal),
            PointReward::Distance => -1.0,
            PointReward::Sparse => if state == goal {1.0} else {0.0},
            PointReward::SparseTimePenalty => if state == goal {1.0} else {-1.0},
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
            PointReward::SparseTimePenalty => (-1.0 * timelimit, 1.0),
        }
    }
}