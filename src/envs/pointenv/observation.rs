use {
    super::{
        super::{
            DistanceMeasure,
            TensorConvertible,
            VectorConvertible,
        },
        line::PointLine,
        state::PointState,
    },
    candle_core::{
        Device,
        Tensor,
    },
};

/// The observation type for the PointEnv environment
///
/// A [PointObs] is a Goal-Aware observation which consists of the current
/// [PointState], the goal [PointState] and a list of [PointLine]s which
/// represent the obstacles in the environment.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PointObs {
    state: PointState,
    goal: PointState,
    obs: Vec<PointLine>,
}

impl From<(PointState, PointState, &[PointLine])> for PointObs {
    /// Convert `(PointState, PointState, &[PointLine])` into a [PointObs]
    fn from(value: (PointState, PointState, &[PointLine])) -> Self {
        Self {
            state: value.0,
            goal: value.1,
            obs: value.2.to_vec(),
        }
    }
}

impl VectorConvertible for PointObs {
    /// Convert a [PointObs] into a [`Vec<f64>`] with preprocessing
    ///
    /// Preprocessing is currently a no-op
    ///
    /// The length of the vector should be `2 + 2 + 4 * n`, where `n` is the
    /// number of walls in the environment.
    ///
    /// Because we cannot know the number of walls in the environment at compile
    /// time, we cannot exactly check the length of the vector. The best we can
    /// do is panic unless the length is at least 4 and is divisible by 4.
    fn from_vec_pp(value: Vec<f64>) -> Self {
        Self::from_vec(value)
    }

    /// Convert a [`Vec<f64>`] into a [PointObs]
    ///
    ///
    /// The length of the vector should be `2 + 2 + 4 * n`, where `n` is the
    /// number of walls in the environment.
    ///
    /// Because we cannot know the number of walls in the environment at compile
    /// time, we cannot exactly check the length of the vector. The best we can
    /// do is panic unless the length is at least 4 and is divisible by 4.
    fn from_vec(value: Vec<f64>) -> Self {
        let state = PointState::from((value[0], value[1]));
        let goal = PointState::from((value[2], value[3]));
        // assert!(value[4..].len() % 4 == 0);
        // let obs: Vec<PointLine> = value[4..]
        //     .chunks(4)
        //     .map(|c| {
        //         PointLine::from((
        //             PointState::from((c[0], c[1])),
        //             PointState::from((c[2], c[3])),
        //         ))
        //     })
        //     .collect();
        let obs = Vec::new();
        Self { state, goal, obs }
    }

    /// Convert a [PointObs] into a [`Vec<f64>`] of the form
    /// `[Sx, Sy, Gx, Gy, Ax, Ay, Bx, By, ...]`
    ///
    /// The length of the vector will be `2 + 2 + 4 * n`, where `n` is the
    /// number of walls in the environment.
    fn to_vec(value: Self) -> Vec<f64> {
        // let mut v = vec![value.state.x(), value.state.y(), value.goal.x(), value.goal.y()];
        // v.extend(value.obs.iter().flat_map(|l| vec![l.A.x(), l.A.y(), l.B.x(), l.B.y()]));
        // v
        vec![
            value.state.x(),
            value.state.y(),
            value.goal.x(),
            value.goal.y(),
        ]
    }
}

impl TensorConvertible for PointObs {
    /// Convert a [PointObs] into a [Tensor] with preprocessing
    ///
    /// Preprocessing is currently a no-op
    ///
    /// This function tries to convert the [Tensor] to a [`Vec<f64>`], which panics if
    /// the [Tensor] is not either 1-dimensional or has a 0-sized batch dimension.
    /// It then passes the result to [`VectorConvertible::from_vec_pp(value: Vec<f64>)`].
    ///
    /// For a detailed description of the preprocessing applied, see
    /// [`VectorConvertible::from_vec_pp(value: Vec<f64>)`].
    fn from_tensor_pp(value: Tensor) -> Self {
        Self::from_tensor(value)
    }

    /// Convert a [Tensor] into a [PointObs]
    ///
    /// This function tries to convert the [Tensor] to a [`Vec<f64>`], which panics if
    /// the [Tensor] is not either 1-dimensional or has a 0-sized batch dimension.
    /// It then passes the result to [`VectorConvertible::from_vec_pp(value: Vec<f64>)`].
    fn from_tensor(value: Tensor) -> Self {
        Self::from_vec(value.to_vec1::<f64>().unwrap())
    }

    /// Convert a [PointObs] into a [Tensor] (with no batch dimension) on
    /// the given device.
    fn to_tensor(
        value: Self,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        Tensor::new(Self::to_vec(value), device)
    }
}

impl DistanceMeasure for PointObs {
    /// The distance between two [PointObs] is the distance between their
    /// current [PointState]s
    fn distance(
        s1: &Self,
        s2: &Self,
    ) -> f64 {
        (s1.state).squared_distance_to(&s2.state).sqrt()
    }
}
