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
    serde::Serialize,
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
    pyo3::PyObject,
    rand::Rng,
    std::ops::RangeInclusive,
};

fn preprocess_action(mut value: Vec<f64>) -> Vec<f64> {
    value[0] = (2.0 * value[0]).clamp(-2.0, 2.0);
    value
}

fn preprocess_state(mut value: Vec<f64>) -> Vec<f64> {
    value[2] /= 8.0;
    value
}

/// A wrapper around the Gymnasium Pendulum-v1 environment originally from OpenAI
///
/// For more details, see the Gymnasium Pendulum-v1
/// [documentation](https://gymnasium.farama.org/environments/classic_control/pendulum/).
pub struct PendulumEnv {
    config: PendulumConfig,
    env: PyObject,
    timelimit: usize,
    current_observation: PendulumState,
    action_space: Vec<usize>,
    observation_space: Vec<usize>,
}

/// The configuration struct for the [PendulumEnv] environment
///
/// The Pendulum environment has no configuration options, so this struct
/// just contains the name of the environment (which is "Pendulum-v1").
#[derive(Clone, Serialize)]
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

/// The action type for the [PendulumEnv] environment
///
/// The Pendulum environment has a single action, which is the torque applied
/// to the free end of the pendulum. This is represented by a single `f64` value:
/// `[tau]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PendulumAction {
    tau: OrderedFloat<f64>,
}
impl Sampleable for PendulumAction {
    /// Sample a random [PendulumAction]
    ///
    /// This returns a [PendulumAction] with a random value for tau within the
    /// range given by the domain.
    ///
    /// This function panics if the number of ranges in the domain is not 1.
    fn sample(
        rng: &mut dyn rand::RngCore,
        domain: &[RangeInclusive<f64>],
    ) -> Self {
        assert!(domain.len() == 1);
        Self {
            tau: OrderedFloat(rng.gen_range(domain[0].clone())),
        }
    }
}
impl VectorConvertible for PendulumAction {
    /// Convert a [`Vec<f64>`] to a [PendulumAction] with preprocessing applied
    ///
    /// Preprocessing consists of multiplying the given value by `2.0` and
    /// then clamping the value to the range `[-2.0, 2.0]`.
    ///
    /// This helps a neural network based agent learn faster because it
    /// essentially normalizes the action space to the neural network
    /// friendly range of `[-1.0, 1.0]`.
    ///
    /// This function panics if the number of elements in the Vec is not 1.
    fn from_vec_pp(value: Vec<f64>) -> Self {
        Self::from_vec(preprocess_action(value))
    }

    /// Convert a [`Vec<f64>`] to a [PendulumAction] without preprocessing
    ///
    /// This function panics if the number of elements in the Vec is not 1.
    fn from_vec(value: Vec<f64>) -> Self {
        assert!(value.len() == 1);
        Self {
            tau: OrderedFloat(value[0]),
        }
    }

    /// Convert a [PendulumAction] to a [`Vec<f64>`] of the form `[tau]`.
    fn to_vec(value: Self) -> Vec<f64> {
        vec![*value.tau]
    }
}
impl TensorConvertible for PendulumAction {
    /// Convert a [Tensor] to a [PendulumAction] with preprocessing applied
    ///
    /// This function tries to convert the [Tensor] to a [`Vec<f64>`], which panics if
    /// the [Tensor] is not either 1-dimensional or has a 0-sized batch dimension.
    /// It then passes the result to [`VectorConvertible::from_vec_pp(value: Vec<f64>)`].
    ///
    /// For a detailed description of the preprocessing applied, see
    /// [`VectorConvertible::from_vec_pp(value: Vec<f64>)`].
    fn from_tensor_pp(value: Tensor) -> Self {
        Self::from_vec_pp(value.to_vec1::<f64>().unwrap())
    }

    /// Convert a [Tensor] to a [PendulumAction] without preprocessing
    ///
    /// This function tries to convert the [Tensor] to a [`Vec<f64>`], which panics if
    /// the [Tensor] is not either 1-dimensional or has a 0-sized batch dimension.
    /// It then passes the result to [`VectorConvertible::from_vec(value: Vec<f64>)`].
    fn from_tensor(value: Tensor) -> Self {
        Self::from_vec(value.to_vec1::<f64>().unwrap())
    }

    /// Convert a [PendulumAction] to a [Tensor] (with no batch dimension) on
    /// the given device.
    fn to_tensor(
        value: Self,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        Tensor::new(Self::to_vec(value), device)
    }
}

/// The observation type for the [PendulumEnv] environment
///
/// The observation for the Pendulum environment consists of the `(x, y)`
/// coordinates of the free end of the pendulum and the angular velocity
/// of the pendulum. These are represented by three `f64` values:
/// `[x, y, velocity]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PendulumState {
    x: OrderedFloat<f64>,
    y: OrderedFloat<f64>,
    velocity: OrderedFloat<f64>,
}
impl VectorConvertible for PendulumState {
    /// Convert a [`Vec<f64>`] to a [PendulumState] with preprocessing
    ///
    /// Because the observation domain is `[-1.0, 1.0]` for the `(x, y)`
    /// coordinates, and `[-8.0, 8.0]` for the velocity, preprocessing
    /// consists of dividing the last value by `8.0`.
    ///
    /// This helps a neural network based agent learn faster because it
    /// essentially normalizes the action space to the neural network
    /// friendly range of `[-1.0, 1.0]`.
    ///
    /// This function panics if the number of elements in the Vec is not 3.
    fn from_vec_pp(value: Vec<f64>) -> Self {
        Self::from_vec(preprocess_state(value))
    }

    /// Convert a [`Vec<f64>`] to a [PendulumState] without preprocessing
    ///
    /// This function panics if the number of elements in the Vec is not 3.
    fn from_vec(value: Vec<f64>) -> Self {
        assert!(value.len() == 3);
        Self {
            x: OrderedFloat(value[0]),
            y: OrderedFloat(value[1]),
            velocity: OrderedFloat(value[2]),
        }
    }

    /// Convert a [PendulumState] to a [`Vec<f64>`] of size 3.
    fn to_vec(value: Self) -> Vec<f64> {
        vec![*value.x, *value.y, *value.velocity]
    }
}
impl TensorConvertible for PendulumState {
    /// Convert a [Tensor] to a [PendulumState] with preprocessing applied
    ///
    /// This function tries to convert the [Tensor] to a [`Vec<f64>`], which panics if
    /// the [Tensor] is not either 1-dimensional or has a 0-sized batch dimension.
    /// It then passes the result to [`VectorConvertible::from_vec_pp(value: Vec<f64>)`].
    ///
    /// For a detailed description of the preprocessing applied, see
    /// [`VectorConvertible::from_vec_pp(value: Vec<f64>)`].
    fn from_tensor_pp(value: Tensor) -> Self {
        Self::from_vec_pp(value.to_vec1::<f64>().unwrap())
    }

    /// Convert a [Tensor] to a [PendulumState] without preprocessing
    ///
    /// This function tries to convert the [Tensor] to a [`Vec<f64>`], which panics if
    /// the [Tensor] is not either 1-dimensional or has a 0-sized batch dimension.
    /// It then passes the result to [`VectorConvertible::from_vec(value: Vec<f64>)`].
    fn from_tensor(value: Tensor) -> Self {
        Self::from_vec(value.to_vec1::<f64>().unwrap())
    }

    /// Convert a [PendulumState] to a Tensor (with no batch dimension) on
    /// the given device.
    fn to_tensor(
        value: Self,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        Tensor::new(Self::to_vec(value), device)
    }
}
impl DistanceMeasure for PendulumState {
    /// Compute the distance between two [PendulumState]s
    ///
    /// The distance between two [PendulumState]s is defined as the 3-dimensional
    /// Euclidean distance between their `(x, y)` coordinates and their velocities.
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

    /// Create a new [PendulumEnv] with the given [PendulumConfig]
    ///
    /// This function panics if the Gymnasium environment cannot be created.
    fn new(config: Self::Config) -> Result<Box<Self>> {
        let (env, action_space, observation_space) = gym_create_env(&config.name, false)?;
        Ok(Box::new(Self {
            config: config.clone(),
            env,
            timelimit: 200,
            current_observation: PendulumState {
                x: OrderedFloat(-1.0),
                y: OrderedFloat(0.0),
                velocity: OrderedFloat(0.0),
            },
            action_space,
            observation_space,
        }))
    }

    /// Reset the environment with the given seed
    fn reset(
        &mut self,
        seed: u64,
    ) -> Result<Self::Observation> {
        self.current_observation = gym_reset_env(&self.env, seed, false)?;
        Ok(self.current_observation())
    }

    /// Step the environment with the given action
    ///
    /// The return type is a [Step] struct, which contains the following fields:
    /// - `observation`: the new observation after the step
    /// - `reward`: the reward for taking the action
    /// - `terminated`: whether the episode is terminated
    /// - `truncated`: whether the episode is truncated
    fn step(
        &mut self,
        action: Self::Action,
    ) -> Result<Step<Self::Observation, Self::Action>> {
        let step: Step<Self::Observation, Self::Action> = gym_step_env(&self.env, action, false)?;
        self.current_observation = step.observation.clone();
        Ok(step)
    }

    /// Return the maximum number of steps allowed before the episode is truncated.
    fn timelimit(&self) -> usize {
        self.timelimit
    }

    /// The action space of [PendulumEnv] is a single `f64` value in the range `[-2.0, 2.0]`.
    fn action_space(&self) -> Vec<usize> {
        self.action_space.clone()
    }

    /// The action domain of [PendulumEnv] is a single range `[-2.0, 2.0]`.
    fn action_domain(&self) -> Vec<RangeInclusive<f64>> {
        vec![-2.0..=2.0]
    }

    /// The observation space of [PendulumEnv] is a 3-dimensional vector of `f64` values.
    fn observation_space(&self) -> Vec<usize> {
        self.observation_space.clone()
    }

    /// The observation domain of [PendulumEnv] is a 3-dimensional vector of ranges:
    /// - `[-1.0, 1.0]`: the `(x, y)` coordinates of the free end of the pendulum
    /// - `[-8.0, 8.0]`: the angular velocity of the pendulum
    fn observation_domain(&self) -> Vec<RangeInclusive<f64>> {
        vec![-1.0..=1.0, -1.0..=1.0, -8.0..=8.0]
    }

    /// Return the current observation of the [PendulumEnv]
    fn current_observation(&self) -> Self::Observation {
        self.current_observation.clone()
    }

    /// Return the value range of the reward function, with a 40% padding on the upper bound.
    fn value_range(&self) -> (f64, f64) {
        // the reward per timestep is in [-16.2736044, 0.0]
        // the environment is reset after 200 timesteps
        let lo: f64 = -16.2736044 * self.timelimit as f64;
        let hi: f64 = 0.0;

        // add 40% padding to upper bound
        let padding = (lo.abs() + hi.abs()) * 0.4;

        (lo, hi + padding)
    }

    /// Return the [PendulumConfig] used to create this [PendulumEnv]
    fn config(&self) -> Self::Config {
        self.config.clone()
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
