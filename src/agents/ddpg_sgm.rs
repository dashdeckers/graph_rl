use {
    super::RunMode,
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
        components::{
            ReplayBuffer,
            sgm::{
                DistanceMode,
                try_adding_node,
            },
        },
        configs::DDPG_SGM_Config,
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
        path::Path,
    },
};


#[allow(non_camel_case_types)]
#[derive(Clone)]
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
    last_waypoint: Option<Env::Observation>,
    try_counter: usize,

    dist_mode: DistanceMode,
    sgm_close_enough: f64,
    sgm_max_tries: usize,
    sgm_maxdist: f64,
    sgm_tau: f64,

    config: DDPG_SGM_Config,
}

impl<'a, Env> DDPG_SGM<'a, Env>
where
    Env: Environment,
    Env::Observation: Clone + Debug + Eq + Hash + TensorConvertible + GoalAwareObservation + DistanceMeasure,
    <Env::Observation as GoalAwareObservation>::State: Clone + Eq + DistanceMeasure,
{
    pub fn new_buffer(&mut self, buffer_capacity: usize) {
        self.ddpg.new_buffer(buffer_capacity);
    }

    fn distance(
        &self,
        from_state: &<Env::Observation as GoalAwareObservation>::State,
        goal_state: &<Env::Observation as GoalAwareObservation>::State,
        from_obs: &<Env::Observation as GoalAwareObservation>::View,
    ) -> f64 {
        match self.dist_mode {
            DistanceMode::True => {
                <<Env as Environment>::Observation as GoalAwareObservation>::State::distance(
                    from_state,
                    goal_state,
                )
            },
            DistanceMode::Estimated => {
                let state = <Env::Observation>::to_tensor(
                    Env::Observation::new(
                        from_state,
                        goal_state,
                        from_obs,
                    ),
                    &self.device,
                ).unwrap();
                self.ddpg.critic_forward_item(
                    &state,
                    &self.ddpg.actor_forward_item(&state).unwrap(),
                ).unwrap().to_vec1::<f64>().unwrap()[0]
            },
        }
    }

    fn get_closest(
        &self,
        goal_state: &<Env::Observation as GoalAwareObservation>::State,
    ) -> Option<Env::Observation> {

        let mut candidate = None;
        let mut min_distance = f64::INFINITY;

        for s2 in self.sgm.node_indices() {
            let s2 = self.sgm.node_weight(s2).unwrap();

            let distance = self.distance(
                s2.achieved_goal(),
                goal_state,
                s2.observation(),
            );

            if distance <= self.sgm_close_enough && (candidate.is_none() || distance < min_distance) {
                candidate = Some(s2.clone());
                min_distance = distance;
            }
        }

        candidate
    }

    fn generate_plan(
        &self,
        obs: &Env::Observation,
    ) -> Vec<Env::Observation> {

        let start = self.get_closest(obs.achieved_goal());
        let goal = self.get_closest(obs.desired_goal());

        if let (Some(start), Some(goal)) = (start, goal) {
            let path = astar(
                &self.sgm,
                self.indices[&start],
                |n| n == self.indices[&goal],
                |e| *e.weight(),
                |_| OrderedFloat(0.0),
            );

            match path {
                Some((_, path)) => path.into_iter().rev().map(|n| self.sgm.node_weight(n).unwrap().clone()).collect(),
                None => Vec::new(),
            }
        } else {
            Vec::new()
        }
    }

    fn splice_goal_into_obs(
        &self,
        obs: &Env::Observation,
        goal: &Env::Observation,
    ) -> Env::Observation {
        Env::Observation::new(
            obs.achieved_goal(),
            goal.desired_goal(),
            obs.observation(),
        )
    }

    fn tensor_is_true(
        &self,
        tensor: &Tensor,
    ) -> bool {
        tensor.to_vec1::<u8>().unwrap().iter().all(|&x| x > 0)
    }

    pub fn from_config_with_ddpg(
        device: &Device,
        config: &DDPG_SGM_Config,
        ddpg: DDPG<'a>,
    ) -> Result<Box<Self>> {
        Ok(Box::new(Self {
            ddpg,
            device: device.clone(),

            sgm: StableGraph::default(),
            indices: HashMap::new(),
            plan: Vec::new(),
            try_counter: 0,

            goal_obs: None,
            last_waypoint: None,
            dist_mode: config.distance_mode,
            sgm_max_tries: config.sgm_max_tries,
            sgm_close_enough: config.sgm_close_enough,
            sgm_maxdist: config.sgm_maxdist,
            sgm_tau: config.sgm_tau,

            config: config.clone(),
        }))
    }

    pub fn save(
        &self,
        path: &dyn AsRef<Path>,
        suffix: &str,
    ) -> Result<()> {
        self.ddpg.save(path, suffix)
    }

    pub fn load(
        &mut self,
        path: &dyn AsRef<Path>,
        suffix: &str,
    ) -> Result<()> {
        self.ddpg.load(path, suffix)
    }
}

impl<'a, Env> Algorithm for DDPG_SGM<'a, Env>
where
    Env: Environment,
    Env::Observation: Clone + Debug + Eq + Hash + TensorConvertible + GoalAwareObservation + DistanceMeasure,
    <Env::Observation as GoalAwareObservation>::State: Clone + Debug + Eq + Hash + TensorConvertible + DistanceMeasure,
{
    type Config = DDPG_SGM_Config;

    fn config(&self) -> &Self::Config {
        &self.config
    }

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
            try_counter: 0,

            goal_obs: None,
            last_waypoint: None,
            dist_mode: config.distance_mode,
            sgm_max_tries: config.sgm_max_tries,
            sgm_close_enough: config.sgm_close_enough,
            sgm_maxdist: config.sgm_maxdist,
            sgm_tau: config.sgm_tau,

            config: config.clone(),
        }))
    }

    fn actions(
        &mut self,
        state: &Tensor,
        mode: RunMode,
    ) -> Result<Tensor> {
        let curr_obs = <Env::Observation>::from_tensor(state.clone());

        // IF the goal has changed OR we have no goal
        //      forget the plan
        //      set the new goal

        if self.goal_obs.is_none() || curr_obs.desired_goal() != self.goal_obs.as_ref().unwrap().desired_goal() {
            self.plan = Vec::new();
            self.goal_obs = Some(curr_obs.clone());
            self.last_waypoint = None;
        }

        // try adding curr_obs to the graph

        if try_adding_node(
            &mut self.sgm,
            &mut self.indices,
            &curr_obs,
            Env::Observation::distance,
            // &|s1: &Env::Observation, s2: &Env::Observation| {
            //     self.distance(
            //         s1.achieved_goal(),
            //         s2.achieved_goal(),
            //         s1.observation(),
            //     )
            // },
            self.sgm_maxdist,
            self.sgm_tau,
        ).is_ok() {
            warn!("Added node to graph: {:#?}", curr_obs);
        };

        // IF we have a plan AND we have been trying too long
        //      remove the edge
        //      forget the plan

        if !self.plan.is_empty() && self.try_counter > self.sgm_max_tries {
            if let Some(last_waypoint) = self.last_waypoint.clone() {
                let edge = self.sgm.find_edge(
                    self.indices[&last_waypoint],
                    self.indices[&self.plan.last().unwrap()],
                ).unwrap();

                self.sgm.remove_edge(edge);
            }

            self.plan = Vec::new();
            self.try_counter = 0;
        }

        // IF we dont have a plan
        //      try making a plan to the goal

        if self.plan.is_empty() {
            self.plan = self.generate_plan(&curr_obs);
        }

        // IF we have a plan
        //      try reaching the next waypoint
        // ELSE
        //      try reaching the goal

        if !self.plan.is_empty() {
            let waypoint_obs = self.splice_goal_into_obs(
                &curr_obs,
                self.plan.last().unwrap(),
            );

            warn!("Aiming for Waypoint: {:#?}", waypoint_obs);
            self.ddpg.actions(&<Env::Observation>::to_tensor(waypoint_obs, &self.device)?, mode)
        } else {
            warn!("Aiming for Goal: {:#?}", curr_obs);
            self.ddpg.actions(state, mode)
        }
    }

    fn train(&mut self) -> Result<()> {
        self.ddpg.train()
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
        if self.plan.is_empty() || self.tensor_is_true(terminated) || self.tensor_is_true(truncated) { // || terminated || truncated
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
            let curr_obs = self.splice_goal_into_obs(
                &<Env::Observation>::from_tensor(state.clone()),
                self.plan.last().unwrap(),
            );
            let next_obs = self.splice_goal_into_obs(
                &<Env::Observation>::from_tensor(next_state.clone()),
                self.plan.last().unwrap(),
            );
            let mut reward = reward.clone();

            warn!("Checking distance between: {:#?} and {:#?}", next_obs.achieved_goal(), next_obs.achieved_goal());

            // Then we check if we have reached the next waypoint
            let distance_to_waypoint = self.distance(
                next_obs.achieved_goal(),
                next_obs.desired_goal(),
                next_obs.observation(),
            );

            // If we have reached the next waypoint, we:
            // - Pop the waypoint from the plan
            // - Reset the try counter
            // - Pretend we got a +1.0 reward from the environment for reaching the waypoint
            if distance_to_waypoint <= self.sgm_close_enough {
                self.last_waypoint = self.plan.pop();
                self.try_counter = 0;
                reward = Tensor::new(vec![1.0], &self.device).unwrap();
            } else {
                self.try_counter += 1;
                warn!("Try counter: {}", self.try_counter);
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








// TODO: maybe we dont need this?



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
                |s1: &Env::Observation, s2: &Env::Observation| {
                    self.distance(
                        s1.achieved_goal(),
                        s2.achieved_goal(),
                        s1.observation(),
                    )
                },
                self.sgm_maxdist,
                self.sgm_tau,
            );
    }
}

