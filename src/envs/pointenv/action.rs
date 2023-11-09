use auto_ops::impl_op_ex;
use candle_core::{Device, Tensor};
use ordered_float::OrderedFloat;
use rand::{Rng, RngCore};

use super::state::PointState;
use super::super::{TensorConvertible, VectorConvertible, Sampleable};

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

// Convert (f64, f64) into PointAction
impl From<(f64, f64)> for PointAction {
    fn from(value: (f64, f64)) -> Self {
        Self {
            dx: OrderedFloat(value.0),
            dy: OrderedFloat(value.1),
        }
    }
}

// Sample a random PointAction
impl Sampleable for PointAction {
    fn sample(
        rng: &mut dyn RngCore,
        domain: &[std::ops::RangeInclusive<f64>]
    ) -> Self {
        debug_assert!(domain.len() == 1);

        let radius = domain[0].end();
        let r: f64 = radius * f64::sqrt(rng.gen_range(0.0..=1.0));
        let theta: f64 = rng.gen_range(0.0..=1.0) * 2.0 * std::f64::consts::PI;

        Self::from((r * theta.cos(), r * theta.sin()))
    }
}

// Convert PointAction from/into Vec<f64>
impl VectorConvertible for PointAction {
    fn from_vec(value: Vec<f64>) -> Self {
        // Make sure the number of elements in the Vec makes sense
        debug_assert!(value.len() == 2);
        Self::from((value[0], value[1]))
    }
    fn to_vec(value: Self) -> Vec<f64> {
        vec![value.dx(), value.dy()]
    }
}

// Convert PointAction from/into Tensor
impl TensorConvertible for PointAction {
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

// Implement helpful operations

// PointState + PointAction AND reference types
impl_op_ex!(+ |p1: &PointState, a: &PointAction| -> PointState {
    PointState::from((
        p1.x() + a.dx(),
        p1.y() + a.dy(),
    ))
});
