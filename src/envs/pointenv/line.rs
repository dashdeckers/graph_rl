#![allow(non_snake_case)]
use {
    super::{
        super::{
            TensorConvertible,
            VectorConvertible,
        },
        state::PointState,
    },
    serde::Serialize,
    candle_core::{
        Device,
        Tensor,
    },
    ordered_float::OrderedFloat,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct PointLine {
    // Let's assume a directionality from A -> B where it makes sense
    pub A: PointState,
    pub B: PointState,
}
impl PointLine {
    /// Check whether the Line contains the point P.
    pub fn contains(
        &self,
        P: &PointState,
    ) -> bool {
        self.A.x().min(self.B.x()) <= P.x()
            && P.x() <= self.A.x().max(self.B.x())
            && self.A.y().min(self.B.y()) <= P.y()
            && P.y() <= self.A.y().max(self.B.y())
    }

    /// Returns the collision point between two lines, if any. \
    /// If the two lines cross, return the collision point and otherwise return None.
    ///
    /// If the two lines are collinear and overlapping, we return the following: \
    /// (self.A --- other.A --- self.B --- other.B) -->  other.A \
    /// (self.A --- other.B --- self.B --- other.A) -->  other.B \
    /// (self.A --- other.A --- other.B --- self.B) -->  other.A \
    /// (self.A --- other.B --- other.A --- self.B) -->  other.B \
    ///
    /// If the two lines are collinear and not overlapping, or parallel, we return None.
    pub fn collision_with(
        &self,
        other: &Self,
    ) -> Option<PointState> {
        let start = self.A;
        let goal = self.B;
        let A = other.A;
        let B = other.B;

        let Gx_Sx = goal.x() - start.x();
        let Gy_Sy = goal.y() - start.y();
        let Bx_Ax = B.x() - A.x();
        let By_Ay = B.y() - A.y();

        let determinant = -Bx_Ax * Gy_Sy + Gx_Sx * By_Ay;

        // if the determinant is zero, we have parallel or collinear lines
        // parallel: have same slope, collinear: on same line
        if determinant.abs() < f64::EPSILON {
            match (self.contains(&other.A), self.contains(&other.B)) {
                (true, true) => {
                    if self.A.squared_distance_to(&other.A) < self.A.squared_distance_to(&other.B) {
                        Some(other.A)
                    } else {
                        Some(other.B)
                    }
                }
                (false, true) => Some(other.B),
                (true, false) => Some(other.A),
                (false, false) => None,
            }
        } else {
            let s = OrderedFloat(
                (-Gy_Sy * (start.x() - A.x()) + Gx_Sx * (start.y() - A.y())) / determinant,
            );

            let t = OrderedFloat(
                (Bx_Ax * (start.y() - A.y()) - By_Ay * (start.x() - A.x())) / determinant,
            );

            let zero = OrderedFloat(0.0);
            let one = OrderedFloat(1.0);

            if s >= zero && s <= one && t >= zero && t <= one {
                Some(PointState::from((
                    start.x() + *(t * Gx_Sx),
                    start.y() + *(t * Gy_Sy),
                )))
            } else {
                None
            }
        }
    }

    /// Bounce back from obstacle in opposite direction, but no further than starting point
    pub fn bounce_from_obstacle(
        position: PointState,
        collision: PointState,
        bounce_factor: OrderedFloat<f64>,
    ) -> PointState {
        // the vector pointing in the opposite direction to the agents movement
        let back_vector = position - collision;

        // normalize it to a unit vector of length 1 and then multiply by bounce_factor
        let normalized_back_vector = back_vector / back_vector.magnitude();
        let bounce_back_vector = bounce_factor * normalized_back_vector;

        // determine the bounced_back state
        let new_state = collision + bounce_back_vector;

        // info!(
        //     concat!(
        //         "\nBackvector: {back_vector:#?},",
        //         "\nNormalized: {normalized_back_vector:#?},",
        //         "\nScaled: {bounce_back_vector:#?}",
        //     ),
        //     back_vector = back_vector,
        //     normalized_back_vector = normalized_back_vector,
        //     bounce_back_vector = bounce_back_vector,
        // );

        // if we would bounce back farther than the original state, we dont bounce
        if PointLine::from((new_state, collision)).contains(&position) {
            position
        } else {
            new_state
        }
    }
}

// Convert (PointState, PointState) into PointLine
impl From<(PointState, PointState)> for PointLine {
    fn from(value: (PointState, PointState)) -> Self {
        Self {
            A: value.0,
            B: value.1,
        }
    }
}

// Convert ((f64, f64), (f64, f64)) into PointLine
impl From<((f64, f64), (f64, f64))> for PointLine {
    fn from(value: ((f64, f64), (f64, f64))) -> Self {
        Self {
            A: PointState::from(value.0),
            B: PointState::from(value.1),
        }
    }
}

// Convert PointLine from/into Vec<f64>
impl VectorConvertible for PointLine {
    fn from_vec(value: Vec<f64>) -> Self {
        // Make sure the number of elements in the Vec makes sense
        debug_assert!(value.len() == 4);
        Self::from((
            PointState::from((value[0], value[1])),
            PointState::from((value[2], value[3])),
        ))
    }
    fn to_vec(value: Self) -> Vec<f64> {
        vec![value.A.x(), value.A.y(), value.B.x(), value.B.y()]
    }
}

// Convert PointLine from/into Tensor
impl TensorConvertible for PointLine {
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
