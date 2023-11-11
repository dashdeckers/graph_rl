use {
    super::{
        gym_wrappers::{
            gym_create_env,
            gym_reset_env,
            gym_step_env,
        },
        DistanceMeasure,
        Environment,
        Renderable,
        Sampleable,
        Step,
        TensorConvertible,
        VectorConvertible,
    },
    anyhow::Result,
    candle_core::{
        Device,
        Tensor,
    },
    egui::Color32,
    egui_plot::{
        Line,
        PlotBounds,
        PlotUi,
    },
    ordered_float::OrderedFloat,
    pyo3::prelude::*,
    rand::Rng,
    std::ops::RangeInclusive,
};

pub struct PendulumEnv {
    env: PyObject,
    current_observation: PendulumState,
    action_space: Vec<usize>,
    observation_space: Vec<usize>,
}

pub struct PendulumConfig {
    name: String,
}
impl Default for PendulumConfig {
    fn default() -> Self {
        Self {
            name: "Pendulum-v1".to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PendulumAction {
    // Torque applied to the free end of the pendulum
    tau: OrderedFloat<f64>,
}
impl Sampleable for PendulumAction {
    fn sample(
        rng: &mut dyn rand::RngCore,
        domain: &[RangeInclusive<f64>],
    ) -> Self {
        debug_assert!(domain.len() == 1);
        Self {
            tau: OrderedFloat(rng.gen_range(domain[0].clone())),
        }
    }
}
impl VectorConvertible for PendulumAction {
    fn from_vec(value: Vec<f64>) -> Self {
        // Make sure the number of elements in the Vec makes sense
        debug_assert!(value.len() == 1);
        Self {
            // Preprocess the action
            tau: OrderedFloat((2.0 * value[0]).clamp(-2.0, 2.0)),
        }
    }
    fn to_vec(value: Self) -> Vec<f64> {
        vec![*value.tau]
    }
}
impl TensorConvertible for PendulumAction {
    fn from_tensor(value: Tensor) -> Self {
        let values = value.squeeze(0).unwrap().to_vec1::<f64>().unwrap();
        Self::from_vec(values)
    }
    fn to_tensor(
        value: Self,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        Tensor::new(Self::to_vec(value), device)
    }
}
// Convert PendulumAction into PyAny
impl IntoPy<PyObject> for PendulumAction {
    fn into_py(
        self,
        py: Python<'_>,
    ) -> PyObject {
        self.tau.into_py(py)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PendulumState {
    // The (x, y) coordinates of the free end of the pendulum
    x: OrderedFloat<f64>,
    y: OrderedFloat<f64>,
    // The angular velocity of the pendulum
    velocity: OrderedFloat<f64>,
}
impl VectorConvertible for PendulumState {
    fn from_vec(value: Vec<f64>) -> Self {
        // Make sure the number of elements in the Vec makes sense
        debug_assert!(value.len() == 3);
        Self {
            x: OrderedFloat(value[0]),
            y: OrderedFloat(value[1]),
            velocity: OrderedFloat(value[2]),
        }
    }
    fn to_vec(value: Self) -> Vec<f64> {
        vec![*value.x, *value.y, *value.velocity]
    }
}
impl TensorConvertible for PendulumState {
    fn from_tensor(value: Tensor) -> Self {
        let values = value.squeeze(0).unwrap().to_vec1::<f64>().unwrap();
        Self::from_vec(values)
    }
    fn to_tensor(
        value: Self,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        Tensor::new(Self::to_vec(value), device)
    }
}
impl DistanceMeasure for PendulumState {
    fn distance(
        s1: &Self,
        s2: &Self,
    ) -> f64 {
        ((s1.x - s2.x).powi(2) + (s1.y - s2.y).powi(2) + (s1.velocity - s2.velocity).powi(2)).sqrt()
    }
}

impl Environment for PendulumEnv {
    type Config = PendulumConfig;
    type Action = PendulumAction;
    type Observation = PendulumState;

    fn new(config: Self::Config) -> Result<Box<Self>> {
        let (env, action_space, observation_space) = gym_create_env(&config.name)?;
        Ok(Box::new(Self {
            env,
            current_observation: PendulumState {
                x: OrderedFloat(-1.0),
                y: OrderedFloat(0.0),
                velocity: OrderedFloat(0.0),
            },
            action_space,
            observation_space,
        }))
    }

    fn reset(
        &mut self,
        seed: u64,
    ) -> Result<Self::Observation> {
        self.current_observation = gym_reset_env(&self.env, seed)?;
        Ok(self.current_observation())
    }

    fn step(
        &mut self,
        action: Self::Action,
    ) -> Result<Step<Self::Observation, Self::Action>> {
        let step: Step<Self::Observation, Self::Action> = gym_step_env(&self.env, action, false)?;
        self.current_observation = step.observation.clone();
        Ok(step)
    }

    fn action_space(&self) -> Vec<usize> {
        self.action_space.clone()
    }

    fn action_domain(&self) -> Vec<RangeInclusive<f64>> {
        vec![-2.0..=2.0]
    }

    fn observation_space(&self) -> Vec<usize> {
        self.observation_space.clone()
    }

    fn observation_domain(&self) -> Vec<RangeInclusive<f64>> {
        vec![-1.0..=1.0, -1.0..=1.0, -8.0..=8.0]
    }

    fn current_observation(&self) -> Self::Observation {
        self.current_observation.clone()
    }

    fn value_range(&self) -> (f64, f64) {
        // the reward per timestep is in [-16.2736044, 0.0]
        // the environment is reset after 200 timesteps
        let lo: f64 = -16.2736044 * 200.0;
        let hi: f64 = 0.0;

        // add 40% padding to upper bound
        let padding = (lo.abs() + hi.abs()) * 0.4;

        (lo, hi + padding)
    }
}

impl Renderable for PendulumEnv {
    fn render(
        &mut self,
        plot_ui: &mut PlotUi,
    ) {
        // Setup plot bounds
        plot_ui.set_plot_bounds(PlotBounds::from_min_max([-1.0, -1.0], [1.0, 1.0]));
        // Draw the Pendulum
        let obs = self.current_observation();
        plot_ui.line(
            Line::new(vec![[0.0, 0.0], [*obs.y, *obs.x]])
                .width(3.0)
                .color(Color32::RED),
        )
    }
}
