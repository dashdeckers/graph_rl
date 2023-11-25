#![allow(unused_imports)]
#![allow(dead_code)]
use {
    super::configs::{
        DDPG_SGM_Config,
        DistanceMode,
    },
    crate::{
        agents::{
            DDPG,
            Algorithm,
            OffPolicyAlgorithm,
            SgmAlgorithm,
        },
        envs::{
            Environment,
            DistanceMeasure,
            TensorConvertible,
            GoalAwareObservation,
        },
        components::sgm,
    },
    candle_core::{
        Device,
        Result,
    },
    ordered_float::OrderedFloat,
    petgraph::{
        stable_graph::StableGraph,
        visit::{
            EdgeRef,
            IntoEdgeReferences,
        },
        Undirected,
    },
    std::{
        fmt::Debug,
        hash::Hash,
    },
};

#[allow(non_camel_case_types)]
pub struct DDPG_SGM<'a, Env>
where
    Env: Environment,
    Env::Observation: Clone + Debug + Eq + Hash + TensorConvertible + GoalAwareObservation,
{
    ddpg: DDPG<'a>,

    sgm: StableGraph<Env::Observation, OrderedFloat<f64>, Undirected>,
    pub plan: Vec<Env::Observation>,
    dist_mode: DistanceMode,

    sgm_freq: usize,
    sgm_maxdist: f64,
    sgm_tau: f64,
}

impl<'a, Env> Algorithm for DDPG_SGM<'a, Env>
where
    Env: Environment,
    Env::Observation: Clone + Debug + Eq + Hash + TensorConvertible + GoalAwareObservation,
{
    type Config = DDPG_SGM_Config;

    fn from_config(
        device: &Device,
        config: &DDPG_SGM_Config,
        size_state: usize,
        size_action: usize,
    ) -> Result<Box<Self>> {
        let ddpg = DDPG::from_config(device, &config.ddpg, size_state, size_action)?;

        Ok(Box::new(Self {
            ddpg: *ddpg,
            sgm: StableGraph::default(),
            plan: Vec::new(),
            dist_mode: config.distance_mode.clone(),
            sgm_freq: config.sgm_freq,
            sgm_maxdist: config.sgm_maxdist,
            sgm_tau: config.sgm_tau,
        }))
    }

    fn actions(
        &mut self,
        state: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor> {



        // if we are failing to reach the next step of the plan:
        //
        //   - search the sgm for a list of candidate observations that are 'close-enough' to the current goal
        //   - if there are some:
        //     - pick the closest candidate observation to the current goal
        //     - generate a plan by A* from the current state to the closest candidate observation
        //     - set the plan
        //   - else:
        //     - set the plan empty
        //
        // if the plan is empty:
        //   - self.ddpg.actions(state)
        // else:
        //   - self.ddpg.actions(plan[0])
        //
        //
        // notes:
        //
        // the current goal is encoded in the state
        //
        // we are failing to reach the next step of the plan if:
        //  - it has been N steps with no progress
        //  - OR we just wait for the episode to truncate
        //
        // candidates that are close enough to the goal are:
        //  - within a certain distance D of the goal
        //  - HOW is this distance determined? it needs to be strict but based on the evolving critic
        //
        // cleanup is a different step, during test-time in the paper
        //  - but it seems like remove the 'faulty' edge can be integrated into the algo above
        //
        // precompute distances (based on the evolving critic || based on true euclidean) with Floyd-Warshall




        self.ddpg.actions(state)
    }

    fn train(&mut self) -> candle_core::Result<()> {
        self.ddpg.train()
    }

    fn run_mode(&self) -> crate::RunMode {
        self.ddpg.run_mode()
    }

    fn set_run_mode(&mut self, run_mode: crate::RunMode) {
        self.ddpg.set_run_mode(run_mode)
    }
}

impl<'a, Env> OffPolicyAlgorithm for DDPG_SGM<'a, Env>
where
    Env: Environment,
    Env::Observation: Clone + Debug + Eq + Hash + TensorConvertible + GoalAwareObservation,
{
    fn remember(
        &mut self,
        state: &candle_core::Tensor,
        action: &candle_core::Tensor,
        reward: &candle_core::Tensor,
        next_state: &candle_core::Tensor,
        terminated: &candle_core::Tensor,
        truncated: &candle_core::Tensor,
    ) {




        // if the plan is NOT empty:
        //
        //   - for both state and next_state, override the encoded goal with plan[0]




        self.ddpg.remember(state, action, reward, next_state, terminated, truncated)
    }

    fn replay_buffer(&self) -> &crate::components::ReplayBuffer {
        self.ddpg.replay_buffer()
    }
}

impl<'a, Env> SgmAlgorithm<Env> for DDPG_SGM<'a, Env>
where
    Env: Environment,
    Env::Observation: Clone + Debug + Eq + Hash + TensorConvertible + GoalAwareObservation + DistanceMeasure,
{
    fn graph(&self) -> &StableGraph<Env::Observation, OrderedFloat<f64>, Undirected> {
        &self.sgm
    }

    fn plan(&self) -> &Vec<Env::Observation> {
        &self.plan
    }

    fn construct_graph(&mut self) {
        self.sgm = self
            .replay_buffer()
            .construct_sgm(
                match self.dist_mode {
                    DistanceMode::True => <Env::Observation>::distance,
                    DistanceMode::Estimated => |_s1, _s2| {
                        // let goal_conditioned_state = s1.clone();
                        // goal_conditioned_state.set_goal_from(s2);
                        // let best_estimated_action = self.ddpg.actor_forward_item(&goal_conditioned_state).unwrap();
                        // let best_estimated_distance = self.ddpg.critic_forward_item(goal_conditioned_state, best_estimated_action).unwrap();
                        // best_estimated_distance.to_vec1::<f64>()[0]
                        todo!()
                    },
                },
                self.sgm_maxdist,
                self.sgm_tau,
            )
            .0;
    }
}

