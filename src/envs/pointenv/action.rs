use {
    super::{
        super::{
            Sampleable,
            TensorConvertible,
            VectorConvertible,
        },
        state::PointState,
    },
    auto_ops::impl_op_ex,
    candle_core::{
        Device,
        Tensor,
    },
    ordered_float::OrderedFloat,
    rand::{
        Rng,
        RngCore,
    },
};

/// The action type for the [`PointEnv`](super::point_env::PointEnv) environment
///
/// A [PointAction] is a 2-dimensional vector of the form `[dx, dy]` which
/// describes the change in position relative to the current position in
/// 2-dimensional space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PointAction {
    dx: OrderedFloat<f64>,
    dy: OrderedFloat<f64>,
}
impl PointAction {
    pub fn dx(&self) -> f64 {
        self.dx.into_inner()
    }

    pub fn dy(&self) -> f64 {
        self.dy.into_inner()
    }

    /// Restrict the [PointAction] to a circle of radius `radius`
    pub fn restrict(
        self,
        radius: f64,
    ) -> Self {
        let zero = PointState::from((0.0, 0.0));
        let step = PointState::from((self.dx(), self.dy()));

        let step_distance = zero.squared_distance_to(&step);

        if OrderedFloat(step_distance) <= OrderedFloat(radius.powi(2)) {
            Self::from((self.dx(), self.dy()))
        } else {
            let ratio = radius / step_distance;
            Self::from((self.dx() * ratio, self.dy() * ratio))
        }
    }
}

impl From<(f64, f64)> for PointAction {
    /// Convert `(f64, f64)` into a [PointAction]
    fn from(value: (f64, f64)) -> Self {
        Self {
            dx: OrderedFloat(value.0),
            dy: OrderedFloat(value.1),
        }
    }
}

impl Sampleable for PointAction {
    /// Sample a random [PointAction]
    ///
    /// The [PointAction] is sampled uniformly from a circle of radius `radius`
    /// which is given by the `end` of the given domain (`start` is assumed
    /// to be 0.0).
    ///
    /// This function panics if the number of ranges in the domain is not 1.
    fn sample(
        rng: &mut dyn RngCore,
        domain: &[std::ops::RangeInclusive<f64>],
    ) -> Self {
        assert!(domain.len() == 1);

        let radius = domain[0].end();
        let r: f64 = radius * f64::sqrt(rng.gen_range(0.0..=1.0));
        let theta: f64 = rng.gen_range(0.0..=1.0) * 2.0 * std::f64::consts::PI;

        Self::from((r * theta.cos(), r * theta.sin()))
    }
}

impl VectorConvertible for PointAction {
    /// Convert a [`Vec<f64>`] into a [PointAction] with preprocessing
    ///
    /// Preprocessing is currently a no-op
    ///
    /// Panics if the Vec does not have exactly 2 elements.
    ///
    /// The elements are assumed to be in the form `[dx, dy]`.
    fn from_vec_pp(value: Vec<f64>) -> Self {
        Self::from_vec(value)
    }

    /// Convert a [`Vec<f64>`] into a [PointAction]
    ///
    /// Panics if the Vec does not have exactly 2 elements.
    ///
    /// The elements are assumed to be in the form `[dx, dy]`.
    fn from_vec(value: Vec<f64>) -> Self {
        assert!(value.len() == 2);
        Self::from((value[0], value[1]))
    }

    /// Convert a [PointAction] into a [`Vec<f64>`] of the form `[dx, dy]`
    fn to_vec(value: Self) -> Vec<f64> {
        vec![value.dx(), value.dy()]
    }
}

impl TensorConvertible for PointAction {
    /// Convert a [Tensor] into a [PointAction] with preprocessing
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

    /// Convert a [Tensor] into a [PointAction]
    ///
    /// This function tries to convert the [Tensor] to a [`Vec<f64>`], which panics if
    /// the [Tensor] is not either 1-dimensional or has a 0-sized batch dimension.
    /// It then passes the result to [`VectorConvertible::from_vec(value: Vec<f64>)`].
    fn from_tensor(value: Tensor) -> Self {
        Self::from_vec(value.to_vec1::<f64>().unwrap())
    }

    /// Convert a [PointAction] to a [Tensor] (with no batch dimension) on
    /// the given device.
    fn to_tensor(
        value: Self,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        Tensor::new(Self::to_vec(value), device)
    }
}

// Implement helpful operations

// PointState + PointAction AND reference types
impl_op_ex!(+ |p1: &PointState, a: &PointAction| -> PointState {
    PointState::from((
        p1.x() + a.dx(),
        p1.y() + a.dy(),
    ))
});
