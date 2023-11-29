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
    },
    candle_core::{
        Device,
        Result,
    },
    ordered_float::OrderedFloat,
    petgraph::{
        stable_graph::{
            StableGraph,
            NodeIndex,
        },
        algo::astar,
        Undirected,
    },
    tracing::warn,
    std::{
        collections::HashMap,
        fmt::Debug,
        hash::Hash,
    },
};


#[allow(non_camel_case_types)]
pub struct DDPG_SGM<'a, Env>
where
    Env: Environment,
    Env::Observation: Clone + Debug + Eq + Hash + TensorConvertible + GoalAwareObservation + DistanceMeasure,
    <Env::Observation as GoalAwareObservation>::State: Clone,
{
    ddpg: DDPG<'a>,
    device: Device,

    sgm: StableGraph<Env::Observation, OrderedFloat<f64>, Undirected>,
    indices: HashMap<Env::Observation, NodeIndex>,
    plan: Vec<Env::Observation>,

    curr_obs: Option<Env::Observation>,
    dist_mode: DistanceMode,
    sgm_close_enough: f64,

    sgm_maxdist: f64,
    sgm_tau: f64,
}

impl<'a, Env> DDPG_SGM<'a, Env>
where
    Env: Environment,
    Env::Observation: Clone + Debug + Eq + Hash + TensorConvertible + GoalAwareObservation + DistanceMeasure,
    <Env::Observation as GoalAwareObservation>::State: Clone + Eq + DistanceMeasure,
{
    fn d(
        dist_mode: &DistanceMode,
        s1: &<Env::Observation as GoalAwareObservation>::State,
        s2: &<Env::Observation as GoalAwareObservation>::State,
    ) -> f64 {
        match dist_mode {
            DistanceMode::True => <Env::Observation as GoalAwareObservation>::State::distance(s1, s2),
            DistanceMode::Estimated => todo!(),
        }
    }

    fn get_closest_to(
        &self,
        state: &<Env::Observation as GoalAwareObservation>::State,
    ) -> Option<Env::Observation> {
        let mut candidate = None;
        let mut min_distance = f64::INFINITY;

        for node in self.sgm.node_indices() {

            // We use the true distances here!! (i.e. Oracle)

            let node = self.sgm.node_weight(node).unwrap();
            let distance = DDPG_SGM::<Env>::d(
                &DistanceMode::True,
                node.achieved_goal(),
                state,
            );

            if distance <= self.sgm_close_enough && (candidate.is_none() || distance < min_distance) {
                candidate = Some(node.clone());
                min_distance = distance;
            }
        }

        candidate
    }

    fn generate_plan(&mut self) {
        if let Some(obs) = &self.curr_obs {
            let start = obs.achieved_goal();
            let goal = obs.desired_goal();

            if let (Some(start), Some(goal)) = (self.get_closest_to(start), self.get_closest_to(goal)) {
                let path = astar(
                    &self.sgm,
                    self.indices[&start],
                    |n| n == self.indices[&goal],
                    |e| *e.weight(),
                    |_| OrderedFloat(0.0),
                );

                self.plan = match path {
                    Some((_, path)) => path.into_iter().map(|n| self.sgm.node_weight(n).unwrap().clone()).collect(),
                    None => Vec::new(),
                };
                return
            }
        }
        self.plan = Vec::new();
    }
}

impl<'a, Env> Algorithm for DDPG_SGM<'a, Env>
where
    Env: Environment,
    Env::Observation: Clone + Debug + Eq + Hash + TensorConvertible + GoalAwareObservation + DistanceMeasure,
    <Env::Observation as GoalAwareObservation>::State: Clone + Debug + Eq + Hash + TensorConvertible + DistanceMeasure,
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
            device: device.clone(),

            sgm: StableGraph::default(),
            indices: HashMap::new(),
            plan: Vec::new(),

            curr_obs: None,
            dist_mode: config.distance_mode.clone(),

            sgm_close_enough: config.sgm_close_enough,
            sgm_maxdist: config.sgm_maxdist,
            sgm_tau: config.sgm_tau,
        }))
    }

    fn actions(
        &mut self,
        state: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor> {
        let new_obs = <Env::Observation>::from_tensor(state.clone());

        // Check if we have not yet initialized the goal.
        // This should only happen at the very beginning of training
        if self.curr_obs.is_none() {
            self.curr_obs = Some(new_obs);
            return self.ddpg.actions(state);
        }

        // Check if the environment has given us a new objective, and if so, update the graph and plan.
        // This should happen at the beginning of each episode except the first
        if new_obs.desired_goal() != self.curr_obs.as_ref().unwrap().desired_goal() {
            self.curr_obs = Some(new_obs.clone());

            self.construct_graph();
            self.generate_plan();
            warn!("New goal: {:#?}", self.curr_obs.as_ref().unwrap().desired_goal());
            warn!("New plan: {:#?}", self.plan.iter().map(|o| o.achieved_goal()).collect::<Vec<_>>());
        }

        // Check if we have already reached the next step of the plan
        if !self.plan.is_empty() {
            let distance_to_waypoint = DDPG_SGM::<Env>::d(
                &self.dist_mode,
                new_obs.achieved_goal(),
                self.plan.last().unwrap().achieved_goal(),
            );

            if distance_to_waypoint <= self.sgm_close_enough {
                let popped = self.plan.pop();
                warn!("Reached waypoint ({:#?})", popped.as_ref().unwrap().achieved_goal());
            }
        }

        // If the plan is now empty, we default to the DDPG policy
        // by passing the current state with the end-goal to the DDPG agent
        if self.plan.is_empty() {
            self.ddpg.actions(state)
        } else {
            // Otherwise we pass the current state with the next waypoint as the goal
            let mut new_obs = new_obs;
            new_obs.set_desired_goal(self.plan.last().unwrap().desired_goal());
            self.ddpg.actions(&<Env::Observation>::to_tensor(new_obs, &self.device)?)
        }
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
    Env::Observation: Clone + Debug + Eq + Hash + TensorConvertible + GoalAwareObservation + DistanceMeasure,
    <Env::Observation as GoalAwareObservation>::State: Clone + Debug + Eq + Hash + TensorConvertible + DistanceMeasure,
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
        self.ddpg.remember(state, action, reward, next_state, terminated, truncated);
    }

    fn replay_buffer(&self) -> &crate::components::ReplayBuffer {
        self.ddpg.replay_buffer()
    }
}

impl<'a, Env> SgmAlgorithm<Env> for DDPG_SGM<'a, Env>
where
    Env: Environment,
    Env::Observation: Clone + Debug + Eq + Hash + TensorConvertible + GoalAwareObservation + DistanceMeasure,
    <Env::Observation as GoalAwareObservation>::State: Clone + Debug + Eq + Hash + TensorConvertible + DistanceMeasure,
{
    fn graph(&self) -> &StableGraph<Env::Observation, OrderedFloat<f64>, Undirected> {
        &self.sgm
    }

    fn plan(&self) -> &Vec<Env::Observation> {
        &self.plan
    }

    fn construct_graph(&mut self) {
        (self.sgm, self.indices) = self
            .replay_buffer()
            .construct_sgm(
                match self.dist_mode {
                    DistanceMode::True => Env::Observation::distance,
                    DistanceMode::Estimated => {
                        todo!()
                    },
                },
                self.sgm_maxdist,
                self.sgm_tau,
            );
    }
}

