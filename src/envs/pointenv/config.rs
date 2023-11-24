use {
    super::{
        line::PointLine,
        reward::PointReward,
    },
    serde::Serialize,
    rand::{
        rngs::StdRng,
        Rng,
        SeedableRng,
    },
};

/// The configuration struct for the PointEnv environment.
///
/// # Fields
/// * `width` - The width of the environment.
/// * `height` - The height of the environment.
/// * `walls` - The walls of the environment given as a Vec of [`PointLines`](super::line::PointLine)
/// * `timelimit` - The maximum number of steps before the episode is truncated.
/// * `step_radius` - The radius that defines the maximum distance the agent can reach in one step.
/// * `term_radius` - If the agent is within this radius of the goal, the episode is terminated.
/// * `bounce_factor` - The percentage of the traveled distance that the agent bounces back when it hits a wall.
/// * `reward` - The reward function. For more information, see [`PointReward`](super::reward::PointReward)
/// * `seed` - The seed for the random number generator.
///
/// # Example
/// ```
/// use graph_rl::envs::{
///     PointEnvConfig,
///     PointReward,
/// };
///
/// let config = PointEnvConfig::default();
/// assert_eq!(config.width, 5);
/// assert_eq!(config.height, 5);
/// assert_eq!(config.walls, None);
/// assert_eq!(config.timelimit, 30);
/// assert_eq!(config.step_radius, 1.0);
/// assert_eq!(config.term_radius, 0.5);
/// assert_eq!(config.bounce_factor, 0.1);
/// assert_eq!(config.reward, PointReward::Distance);
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct PointEnvConfig {
    pub width: usize,
    pub height: usize,
    pub walls: Option<Vec<PointLine>>,
    pub timelimit: usize,
    pub step_radius: f64,
    pub term_radius: f64,
    pub bounce_factor: f64,
    pub reward: PointReward,
    pub seed: u64,
}
impl Default for PointEnvConfig {
    fn default() -> Self {
        Self {
            width: 5,
            height: 5,
            walls: None,
            timelimit: 30,
            step_radius: 1.0,
            term_radius: 0.5,
            bounce_factor: 0.1,
            reward: PointReward::Distance,
            seed: StdRng::from_entropy().gen::<u64>(),
        }
    }
}
impl PointEnvConfig {
    /// Creates a new PointEnvConfig.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: usize,
        height: usize,
        walls: Option<Vec<PointLine>>,
        timelimit: usize,
        step_radius: f64,
        term_radius: f64,
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
            bounce_factor,
            reward,
            seed,
        }
    }
}
