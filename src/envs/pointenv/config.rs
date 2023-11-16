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
