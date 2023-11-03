use super::state::PointState;
use super::super::DistanceMeasure;

#[derive(Debug)]
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
            PointReward::Euclidean => -PointState::distance(state, goal),
            PointReward::Sparse => if state == goal {1.0} else {0.0},
            PointReward::SparseTimePenalty => if state == goal {1.0} else {-1.0},
        }
    }
}