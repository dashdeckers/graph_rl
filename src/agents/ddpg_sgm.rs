#![allow(unused_imports)]
#![allow(dead_code)]
use {
    super::configs::DDPG_SGM_Config,
    crate::{
        agents::{
            DDPG,
            Algorithm,
            OffPolicyAlgorithm,
        },
        envs::{
            Environment,
            DistanceMeasure,
            TensorConvertible,
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
    Env::Observation: Debug + Clone + Eq + Hash + TensorConvertible,
{
    ddpg: DDPG<'a>,
    sgm: StableGraph<Env::Observation, OrderedFloat<f64>, Undirected>,

    sgm_freq: usize,
    sgm_maxdist: f64,
    sgm_tau: f64,
}

impl<'a, Env> Algorithm for DDPG_SGM<'a, Env>
where
    Env: Environment,
    Env::Observation: Debug + Clone + Eq + Hash + TensorConvertible,
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
            sgm_freq: config.sgm_freq,
            sgm_maxdist: config.sgm_maxdist,
            sgm_tau: config.sgm_tau,
        }))
    }

    fn actions(
        &mut self,
        state: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor> {
        // here we get a goal-aware state
        // if we dont already have a plan, we need to generate one
        // query the sgm for the nearest state to the goal (either via a helper function or select goal from sgm)
        // query a shortest path algorithm to generate a plan using the sgm. save this plan.
        // if the last action did not succesfully result in the next step of the plan, perform cleanup
        // remove that edge from the sgm
        // generate a new plan
        // query the actor for the next action according to the goal given by the plan
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
    Env::Observation: Debug + Clone + Eq + Hash + TensorConvertible,
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
        // probably do something like HER here, just store the transitions with the goal it was conditioned on
        self.ddpg.remember(state, action, reward, next_state, terminated, truncated)
    }

    fn replay_buffer(&self) -> &crate::components::ReplayBuffer {
        self.ddpg.replay_buffer()
    }
}