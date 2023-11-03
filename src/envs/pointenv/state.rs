use auto_ops::impl_op_ex;
use candle_core::{Device, Tensor};
use ordered_float::OrderedFloat;
use rand::{Rng, RngCore};

use super::super::{TensorConvertible, VectorConvertible, DistanceMeasure};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    pub fn sample(
        rng: &mut dyn RngCore,
        width: f64,
        height: f64,
    ) -> Self {
        Self::from((rng.gen_range(0.0..=width), rng.gen_range(0.0..=height)))
    }

    pub fn restrict(
        self,
        width: f64,
        height: f64,
    ) -> Self {
        Self::from((self.x().clamp(0.0, width), self.y().clamp(0.0, height)))
    }

    pub fn squared_distance_to(
        &self,
        other: &Self,
    ) -> f64 {
        let dx = self.x() - other.x();
        let dy = self.y() - other.y();
        dx.powi(2) + dy.powi(2)
    }

    pub fn in_radius_of(
        &self,
        other: &Self,
        radius: f64,
    ) -> bool {
        OrderedFloat(self.squared_distance_to(other)) <= OrderedFloat(radius.powi(2))
    }

    pub fn magnitude(&self) -> f64 {
        self.squared_distance_to(&PointState::from((0.0, 0.0)))
            .sqrt()
    }
}

// Convert (f64, f64) into PointState
impl From<(f64, f64)> for PointState {
    fn from(value: (f64, f64)) -> Self {
        Self {
            x: OrderedFloat(value.0),
            y: OrderedFloat(value.1),
        }
    }
}

// Convert PointState from/into Vec<f64>
impl VectorConvertible for PointState {
    fn from_vec(value: Vec<f64>) -> Self {
        // Make sure the number of elements in the Vec makes sense
        debug_assert!(value.len() == 2);
        Self::from((value[0], value[1]))
    }
    fn to_vec(value: Self) -> Vec<f64> {
        vec![value.x(), value.y()]
    }
}

// Convert PointState from/into Tensor
impl TensorConvertible for PointState {
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

// PointState has a notion of distance
impl DistanceMeasure for PointState {
    fn distance(s1: &Self, s2: &Self) -> f64 {
        s1.squared_distance_to(s2).sqrt()
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
