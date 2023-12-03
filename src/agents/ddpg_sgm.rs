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
        components::ReplayBuffer,
        RunMode,
    },
    candle_core::{
        Device,
        Result,
        Tensor,
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

    goal_obs: Option<Env::Observation>,
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
        if let Some(obs) = &self.goal_obs {
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
                    Some((_, path)) => path.into_iter().rev().map(|n| self.sgm.node_weight(n).unwrap().clone()).collect(),
                    None => Vec::new(),
                };
                return
            }
        }
        self.plan = Vec::new();
    }

    fn splice_states(
        &self,
        state_is_state_from: &Env::Observation,
        goal_is_state_from: &Env::Observation,
    ) -> Env::Observation {
        let mut obs = state_is_state_from.clone();
        obs.set_desired_goal(goal_is_state_from.achieved_goal());
        obs
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

            goal_obs: None,
            dist_mode: config.distance_mode.clone(),

            sgm_close_enough: config.sgm_close_enough,
            sgm_maxdist: config.sgm_maxdist,
            sgm_tau: config.sgm_tau,
        }))
    }

    fn actions(
        &mut self,
        state: &Tensor,
    ) -> Result<Tensor> {
        let curr_obs = <Env::Observation>::from_tensor(state.clone());

        // Check if we have not yet initialized the goal.
        // This should only happen at the very beginning of training
        if self.goal_obs.is_none() {
            self.goal_obs = Some(curr_obs);
            return self.ddpg.actions(state);
        }

        // Check if the environment has given us a new objective, and if so, update the graph and plan.
        // This should happen at the beginning of each episode except the first
        if curr_obs.desired_goal() != self.goal_obs.as_ref().unwrap().desired_goal() {
            self.goal_obs = Some(curr_obs.clone());

            self.construct_graph();
            self.generate_plan();
            warn!("New goal: {:#?}", self.goal_obs.as_ref().unwrap().desired_goal());
            warn!("New plan: {:#?}", self.plan.iter().map(|o| o.achieved_goal()).collect::<Vec<_>>());
        }

        // If the plan is empty, we default to the DDPG policy
        if self.plan.is_empty() {
            self.ddpg.actions(state)
        } else {
            // Otherwise we pass the current state with the next waypoint as the goal
            let waypoint_obs = self.splice_states(
                &curr_obs,
                self.plan.last().unwrap(),
            );
            warn!("Aiming for Waypoint: {:#?}", waypoint_obs);
            self.ddpg.actions(&<Env::Observation>::to_tensor(waypoint_obs, &self.device)?)
        }
    }

    fn train(&mut self) -> Result<()> {
        self.ddpg.train()
    }

    fn run_mode(&self) -> RunMode {
        self.ddpg.run_mode()
    }

    fn set_run_mode(&mut self, run_mode: RunMode) {
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
        state: &Tensor,
        action: &Tensor,
        reward: &Tensor,
        next_state: &Tensor,
        terminated: &Tensor,
        truncated: &Tensor,
    ) {
        // If the plan is empty, we default to the DDPG policy
        if self.plan.is_empty() {
            self.ddpg.remember(
                state,
                action,
                reward,
                next_state,
                terminated,
                truncated,
            );
        } else {
            // Otherwise, we relabel the goals of the state and next_state
            // to reflect that we are trying to reach the next waypoint
            let waypoint = self.plan.last().unwrap();
            let curr_obs = self.splice_states(
                &<Env::Observation>::from_tensor(state.clone()),
                waypoint,
            );
            let next_obs = self.splice_states(
                &<Env::Observation>::from_tensor(next_state.clone()),
                waypoint,
            );
            let mut reward = reward.clone();

            warn!("Checking distance between: {:#?} and {:#?}", next_obs.achieved_goal(), waypoint.achieved_goal());

            // Then we check if we have reached the next waypoint
            let distance_to_waypoint = DDPG_SGM::<Env>::d(
                &self.dist_mode,
                next_obs.achieved_goal(),
                waypoint.achieved_goal(),
            );

            warn!("Distance is: {:#?}", distance_to_waypoint);

            // If we have reached the next waypoint, we:
            // - Pop the waypoint from the plan
            // - Pretend we got a +0.0 reward from the environment for reaching the waypoint
            if distance_to_waypoint <= self.sgm_close_enough {
                warn!("Reached waypoint ({:#?})", waypoint.achieved_goal());

                self.plan.pop();
                reward = Tensor::new(vec![0.0], &self.device).unwrap();
            }

            self.ddpg.remember(
                &<Env::Observation>::to_tensor(curr_obs, &self.device).unwrap(),
                action,
                &reward,
                &<Env::Observation>::to_tensor(next_obs, &self.device).unwrap(),
                terminated,
                truncated,
            );
        }
    }

    fn replay_buffer(&self) -> &ReplayBuffer {
        self.ddpg.replay_buffer()
    }
}

impl<'a, Env> SgmAlgorithm<Env> for DDPG_SGM<'a, Env>
where
    Env: Environment,
    Env::Observation: Clone + Debug + Eq + Hash + TensorConvertible + GoalAwareObservation + DistanceMeasure,
    <Env::Observation as GoalAwareObservation>::State: Clone + Debug + Eq + Hash + TensorConvertible + DistanceMeasure,
{
    fn set_from_config(&mut self, config: &Self::Config) {
        self.dist_mode = config.distance_mode;
        self.sgm_close_enough = config.sgm_close_enough;
        self.sgm_maxdist = config.sgm_maxdist;
        self.sgm_tau = config.sgm_tau;
    }

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
                    DistanceMode::True => |s1: &Env::Observation, s2: &Env::Observation| {
                        DDPG_SGM::<Env>::d(
                            &DistanceMode::True,
                            s1.achieved_goal(),
                            s2.achieved_goal(),
                        )
                    },
                    DistanceMode::Estimated => |s1: &Env::Observation, s2: &Env::Observation| {
                        DDPG_SGM::<Env>::d(
                            &DistanceMode::Estimated,
                            s1.achieved_goal(),
                            s2.achieved_goal(),
                        )
                    },
                },
                self.sgm_maxdist,
                self.sgm_tau,
            );
    }
}

