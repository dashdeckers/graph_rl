use rand::{SeedableRng, Rng, rngs::StdRng};

use super::line::PointLine;
use super::reward::PointReward;

#[derive(Debug)]
pub struct PointEnvConfig {
    pub width: usize,
    pub height: usize,
    pub walls: Option<Vec<PointLine>>,
    pub timelimit: usize,
    pub step_radius: f64,
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
            timelimit: 10,
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
