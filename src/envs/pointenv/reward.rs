use {
    super::{
        super::DistanceMeasure,
        line::PointLine,
        point_env::reachable,
        state::PointState,
    },
    serde::Serialize,
};


/// The reward type for the [`PointEnv`](super::point_env::PointEnv) environment
///
/// The possible reward functions are:
/// - Euclidean: The negative Euclidean distance to the goal
/// - Distance: -1 if the goal is not reachable, 0 otherwise
/// - Sparse: 1 if the goal is reachable, 0 otherwise
///
/// # Examples
///
/// ```
/// use candle_core::Tensor;
/// use candle_envs::pointenv::{
///     PointReward,
///     PointState,
/// };
///
/// let reward_mode = PointReward::Euclidean;
/// let state = PointState::from((0.0, 0.0));
/// let goal = PointState::from((1.0, 1.0));
/// let term_radius = 1.0;
/// let walls = vec![];
/// let timelimit = 10;
/// let width = 10;
/// let height = 10;
///
/// assert_eq!(
///     reward_mode.compute(&state, &goal, term_radius, &walls),
///     -2.0_f64.sqrt(),
/// );
/// assert_eq!(
///     reward_mode.value_range(timelimit, width, height),
///     (-(200.0_f64.sqrt() * 10.0), 0.0)
/// );
///
/// let reward_mode = PointReward::Distance;
/// let state = PointState::from((0.0, 0.0));
/// let goal = PointState::from((0.5, 0.5));
/// let term_radius = 1.0;
/// let walls = vec![];
/// let timelimit = 10;
/// let width = 10;
/// let height = 10;
///
/// assert_eq!(
///     reward_mode.compute(&state, &goal, term_radius, &walls),
///     0.0,
/// );
/// assert_eq!(
///     reward_mode.value_range(timelimit, width, height),
///     (-1.0 * 10.0, 0.0)
/// );
/// ```
///
#[derive(Debug, Clone, Serialize)]
pub enum PointReward {
    Euclidean,
    Distance,
    Sparse,
}
impl PointReward {
    /// Compute the reward for the given state
    ///
    /// For more details, see the documentation of [`PointReward`]
    pub fn compute(
        &self,
        state: &PointState,
        goal: &PointState,
        term_radius: f64,
        walls: &[PointLine],
    ) -> f64 {
        match self {
            PointReward::Euclidean => -PointState::distance(state, goal),
            PointReward::Distance => {
                if reachable(state, goal, term_radius, walls) {
                    0.0
                } else {
                    -1.0
                }
            }
            PointReward::Sparse => {
                if reachable(state, goal, term_radius, walls) {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Compute the value range that the reward can take
    ///
    /// For more details, see the documentation of [`PointReward`]
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
