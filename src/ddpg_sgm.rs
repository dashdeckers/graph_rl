#![allow(unused_imports)]
#![allow(dead_code)]
use std::hash::Hash;
use std::fmt::Debug;

use crate::{
    ddpg::DDPG,
    envs::{
        TensorConvertible,
        DistanceMeasure,
    },
    TrainingConfig,
    sgm,
};
use anyhow::Result;
use ordered_float::OrderedFloat;
use petgraph::{Undirected, stable_graph::StableGraph};
use candle_core::Device;


#[allow(non_camel_case_types)]
pub struct DDPG_SGM<'a, S>
where
    S: Debug + Clone + Eq + Hash + TensorConvertible,
{
    ddpg: DDPG<'a>,
    sgm: StableGraph<S, OrderedFloat<f64>, Undirected>,

    sgm_freq: usize,
    sgm_maxdist: f64,
    sgm_tau: f64,
}

impl <'a, S> DDPG_SGM<'a, S>
where
    S: Debug + Clone + Eq + Hash + TensorConvertible,
{
    pub fn from_config(
        device: &Device,
        config: &TrainingConfig,
        size_state: usize,
        size_action: usize,
    ) -> Result<Self> {
        let ddpg = DDPG::from_config(
            device,
            config,
            size_state,
            size_action,
        )?;

        Ok(Self {
            ddpg,
            sgm: StableGraph::default(),
            sgm_freq: config.sgm_freq,
            sgm_maxdist: config.sgm_maxdist,
            sgm_tau: config.sgm_tau,
        })
    }

    // pub fn actions
        // here we get a goal-aware state
        // if we dont already have a plan, we need to generate one
            // query the sgm for the nearest state to the goal (either via a helper function or select goal from sgm)
            // query a shortest path algorithm to generate a plan using the sgm. save this plan.
        // if the last action did not succesfully result in the next step of the plan, perform cleanup
            // remove that edge from the sgm
            // generate a new plan
        // query the actor for the next action according to the goal given by the plan

    // pub fn remember
        // probably do something like HER here, where we pretend the goal was what the next step in the plan was?

    // pub fn train
        // same same
}


// 