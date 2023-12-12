use {
    super::super::{
        DistanceMeasure,
        Sampleable,
        TensorConvertible,
        VectorConvertible,
    },
    serde::Serialize,
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

/// The state type for the [`PointEnv`](super::point_env::PointEnv) environment
///
/// A [PointState] is a 2-dimensional vector of the form `[x, y]` which describes
/// the position in 2-dimensional space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct PointState {
    x: OrderedFloat<f64>,
    y: OrderedFloat<f64>,
}
impl PointState {
    pub fn x(&self) -> f64 {
        self.x.into_inner()
    }

    pub fn y(&self) -> f64 {
        self.y.into_inner()
    }

    /// Restrict the [PointState] to a rectangle of size `width` x `height`
    pub fn restrict(
        self,
        width: f64,
        height: f64,
    ) -> Self {
        Self::from((self.x().clamp(0.0, width), self.y().clamp(0.0, height)))
    }

    /// Calculate the squared distance to another [PointState]
    pub fn distance_to(
        &self,
        other: &Self,
    ) -> f64 {
        let v = self - other;
        v.x().hypot(v.y())
    }

    /// Check if the [PointState] is within a circle of radius `radius` around
    /// another [PointState]
    pub fn in_radius_of(
        &self,
        other: &Self,
        radius: f64,
    ) -> bool {
        self.distance_to(other) <= radius
    }

    /// Calculate the magnitude of the [PointState] vector
    pub fn magnitude(&self) -> f64 {
        self.x().hypot(self.y())
    }
}

impl From<(f64, f64)> for PointState {
    /// Convert `(f64, f64)` into a [PointState]
    fn from(value: (f64, f64)) -> Self {
        Self {
            x: OrderedFloat(value.0),
            y: OrderedFloat(value.1),
        }
    }
}

impl Sampleable for PointState {
    /// Sample a random [PointState] within a rectangle given by the domain
    ///
    /// The domain is given by two ranges, one for the x-axis and one for the
    /// y-axis.
    ///
    /// This function panics if the number of ranges in the domain is not 2.
    fn sample(
        rng: &mut dyn RngCore,
        domain: &[std::ops::RangeInclusive<f64>],
    ) -> Self {
        assert!(domain.len() == 2);
        Self::from((
            rng.gen_range(domain[0].clone()),
            rng.gen_range(domain[1].clone()),
        ))
    }
}

impl VectorConvertible for PointState {
    /// Convert a [PointState] into a [`Vec<f64>`]
    ///
    /// Preprocessing is currently a no-op
    ///
    /// Panics if the Vec does not have exactly 2 elements.
    ///
    /// The elements are assumed to be in the form `[x, y]`.
    fn from_vec_pp(value: Vec<f64>) -> Self {
        Self::from_vec(value)
    }

    /// Convert a [`Vec<f64>`] into a [PointState]
    ///
    /// Panics if the Vec does not have exactly 2 elements.
    ///
    /// The elements are assumed to be in the form `[x, y]`.
    fn from_vec(value: Vec<f64>) -> Self {
        assert!(value.len() == 2);
        Self::from((value[0], value[1]))
    }

    /// Convert a [PointState] into a [`Vec<f64>`] of the form `[x, y]`
    fn to_vec(value: Self) -> Vec<f64> {
        vec![value.x(), value.y()]
    }
}

impl TensorConvertible for PointState {
    /// Convert a [Tensor] into a [PointState] with preprocessing
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

    /// Convert a [Tensor] into a [PointState]
    ///
    /// This function tries to convert the [Tensor] to a [`Vec<f64>`], which panics if
    /// the [Tensor] is not either 1-dimensional or has a 0-sized batch dimension.
    /// It then passes the result to [`VectorConvertible::from_vec(value: Vec<f64>)`].
    fn from_tensor(value: Tensor) -> Self {
        Self::from_vec(value.to_vec1::<f64>().unwrap())
    }

    /// Convert a [PointState] to a [Tensor] (with no batch dimension) on
    /// the given device.
    fn to_tensor(
        value: Self,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        Tensor::new(Self::to_vec(value), device)
    }
}

impl DistanceMeasure for PointState {
    /// The distance between two [PointState]'s is given by the Euclidean distance.
    fn distance(
        s1: &Self,
        s2: &Self,
    ) -> f64 {
        s1.distance_to(s2)
    }
}

// Implement helpful operations

// PointState + PointState AND reference types
impl_op_ex!(+ |p1: &PointState, p2: &PointState| -> PointState {
    PointState {
        x: p1.x + p2.x,
        y: p1.y + p2.y,
    }
});
// PointState - PointState AND reference types
impl_op_ex!(-|p1: &PointState, p2: &PointState| -> PointState {
    PointState {
        x: p1.x - p2.x,
        y: p1.y - p2.y,
    }
});
// PointState * f64 AND reference types
impl_op_ex!(*|p1: &PointState, s: &f64| -> PointState {
    PointState {
        x: p1.x * s,
        y: p1.y * s,
    }
});
// f64 * PointState AND reference types
impl_op_ex!(*|s: &f64, p1: &PointState| -> PointState {
    PointState {
        x: p1.x * s,
        y: p1.y * s,
    }
});
// PointState * OrderedFloat(f64) AND reference types
impl_op_ex!(*|p1: &PointState, s: &OrderedFloat<f64>| -> PointState {
    PointState {
        x: p1.x * s,
        y: p1.y * s,
    }
});
// OrderedFloat(f64) * PointState AND reference types
impl_op_ex!(*|s: &OrderedFloat<f64>, p1: &PointState| -> PointState {
    PointState {
        x: p1.x * s,
        y: p1.y * s,
    }
});
// PointState / f64 AND reference types
impl_op_ex!(/ |p1: &PointState, s: &f64| -> PointState {
    if OrderedFloat(*s) == OrderedFloat(0.0) {
        panic!("Division by zero is not allowed");
    }
    PointState {
        x: p1.x / s,
        y: p1.y / s,
    }
});
// PointState / OrderedFloat(f64) AND reference types
impl_op_ex!(/ |p1: &PointState, s: &OrderedFloat<f64>| -> PointState {
    if *s == OrderedFloat(0.0) {
        panic!("Division by zero is not allowed");
    }
    PointState {
        x: p1.x / s,
        y: p1.y / s,
    }
});
