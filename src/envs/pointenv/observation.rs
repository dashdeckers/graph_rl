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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PointObs {
    state: PointState,
    goal: PointState,
    obs: Vec<PointLine>,
}

// Convert (PointState, PointState, &[PointLine]) into PointObs
impl From<(PointState, PointState, &[PointLine])> for PointObs {
    fn from(value: (PointState, PointState, &[PointLine])) -> Self {
        Self {
            state: value.0,
            goal: value.1,
            obs: value.2.to_vec(),
        }
    }
}

// Convert PointObs from/into Vec<f64>
impl VectorConvertible for PointObs {
    fn from_vec_pp(_: Vec<f64>) -> Self {
        todo!()
    }
    fn from_vec(value: Vec<f64>) -> Self {
        let state = PointState::from((value[0], value[1]));
        let goal = PointState::from((value[2], value[3]));
        // debug_assert!(value[4..].len() % 4 == 0);
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

// Convert PointObs from/into Tensor
impl TensorConvertible for PointObs {
    fn from_tensor_pp(_: Tensor) -> Self {
        todo!()
    }
    fn from_tensor(value: Tensor) -> Self {
        Self::from_vec(value.to_vec1::<f64>().unwrap())
    }
    fn to_tensor(
        value: Self,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        Tensor::new(Self::to_vec(value), device)
    }
}

// Delegate the notion of distance to the distance between states
impl DistanceMeasure for PointObs {
    fn distance(
        s1: &Self,
        s2: &Self,
    ) -> f64 {
        (s1.state).squared_distance_to(&s2.state).sqrt()
    }
}
