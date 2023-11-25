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
        components::sgm::dot,
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

fn tensor_with_goal<Env>(
    state: &Env::Observation,
    goal: &Env::Observation,
    device: &Device,
) -> Result<candle_core::Tensor>
where
    Env: Environment,
    Env::Observation: Clone + TensorConvertible + GoalAwareObservation,
{
    let mut state = state.clone();
    state.set_desired_goal(goal.desired_goal());
    <Env::Observation>::to_tensor(state, device)
}


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

    state: Option<Env::Observation>,
    goal: Option<Env::Observation>,
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

    fn get_closest_to(&self, obs: Option<Env::Observation>) -> Option<Env::Observation> {
        let mut candidate = None;
        let mut min_distance = f64::INFINITY;

        if let Some(obs) = obs {
            for node in self.sgm.node_indices() {

                // We use the true distances here!! (i.e. Oracle)

                let node = self.sgm.node_weight(node).unwrap();
                let distance = DDPG_SGM::<Env>::d(&DistanceMode::True, node.achieved_goal(), obs.desired_goal());

                if distance <= self.sgm_close_enough && (candidate.is_none() || distance < min_distance) {
                    candidate = Some(node.clone());
                    min_distance = distance;
                    warn!("At distance: {:?}, found candidate: {:?}", min_distance, candidate);
                }
            }
        }

        candidate
    }

    fn generate_plan(&mut self) {
        if let (Some(goal), Some(start)) = (self.get_closest_to(self.goal.clone()), self.get_closest_to(self.state.clone())) {
            warn!("Generating plan from {:?} to {:?}", start, goal);
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
            }
        }
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

            state: None,
            goal: None,
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
        // let state_obs = <Env::Observation>::from_tensor(state.clone());

        // if let Some(goal) = &self.goal.clone() {
        //     if state_obs.desired_goal() != goal.desired_goal() {
        //         // This should only happen at the beginning of each episode
        //         self.goal = Some(state_obs.clone());
        //         self.construct_graph();
        //         self.generate_plan();
        //         self.ddpg.actions(&tensor_with_goal::<Env>(&state_obs, goal, &self.device)?)
        //     } else {
        //         // This should happen at every step
        //         if self.plan.is_empty() {
        //             // Without a plan, we default to the DDPG policy
        //             self.ddpg.actions(state)
        //         } else {
        //             let distance_to_waypoint = DDPG_SGM::<Env>::d(
        //                 &self.dist_mode,
        //                 state_obs.achieved_goal(),
        //                 self.plan.last().unwrap().achieved_goal(),
        //             );
        //             if distance_to_waypoint <= self.sgm_close_enough {
        //                 // If we have reached the next step of the plan, we pop it
        //                 self.plan.pop();
        //             }

        //             if self.plan.is_empty() {
        //                 // If the plan is now empty, we default to the DDPG policy
        //                 self.ddpg.actions(state)
        //             } else {
        //                 // Otherwise, we continue to follow the plan
        //                 self.ddpg.actions(&tensor_with_goal::<Env>(&state_obs, self.plan.last().unwrap(), &self.device)?)
        //             }
        //         }
        //     }
        // } else {
        //     // This should only happen at the very beginning of training
        //     self.goal = Some(state_obs);
        //     self.ddpg.actions(state)
        // }


        // Check if we have not yet initialized the goal.
        // This should only happen at the very beginning of training
        if self.goal.is_none() {
            self.goal = Some(<Env::Observation>::from_tensor(state.clone()));
            warn!("Initialized goal: {:?}", self.goal);
            return self.ddpg.actions(state);
        }

        let state_obs = <Env::Observation>::from_tensor(state.clone());

        // Check if the environment has given us a new objective, and if so, update the graph and plan
        // This should happen at the beginning of each episode
        if state_obs.desired_goal() != self.goal.as_ref().unwrap().desired_goal() {
            self.goal = Some(state_obs.clone());
            self.construct_graph();
            // warn!("Graph: {}", dot(&self.sgm));
            self.generate_plan();
            warn!("New goal: {:?}", self.goal);
            warn!("New plan: {:?}", self.plan);
        }

        // Check if we have already reached the next step of the plan
        if !self.plan.is_empty() {
            let distance_to_waypoint = DDPG_SGM::<Env>::d(
                &self.dist_mode,
                state_obs.achieved_goal(),
                self.plan.last().unwrap().achieved_goal(),
            );
            // warn!("Distance to waypoint: {:?}", distance_to_waypoint);
            if distance_to_waypoint <= self.sgm_close_enough {
                let popped = self.plan.pop();
                warn!("Reached waypoint ({:?}), new plan: {:?}", popped, self.plan);
            }
        }

        // If the plan is now empty, we default to the DDPG policy, otherwise, we continue to follow the plan
        if self.plan.is_empty() {
            self.ddpg.actions(state)
        } else {
            self.ddpg.actions(&tensor_with_goal::<Env>(&state_obs, self.plan.last().unwrap(), &self.device)?)
        }

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



        // self.ddpg.actions(state)
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


        // if the plan is NOT empty:
        //
        //   - for both state and next_state, override the encoded goal with plan[0]

        self.state = Some(<Env::Observation>::from_tensor(state.clone()));

        // if !self.plan.is_empty() {
        //     let state_obs = <Env::Observation>::from_tensor(state.clone());
        //     let next_state_obs = <Env::Observation>::from_tensor(next_state.clone());

        //     // We use the true distances here!! (i.e. Oracle, or as if this was the Environment talking)

        //     let distance_to_waypoint = DDPG_SGM::<Env>::d(
        //         &DistanceMode::True,
        //         next_state_obs.achieved_goal(),
        //         self.plan.last().unwrap().achieved_goal(),
        //     );
        //     let (reward, terminated) = if distance_to_waypoint <= self.sgm_close_enough {
        //         (0.0, true)
        //     } else {
        //         (-1.0, false)
        //     };

        //     self.ddpg.remember(
        //         &tensor_with_goal::<Env>(&state_obs, self.plan.last().unwrap(), &self.device).unwrap(),
        //         action,
        //         &Tensor::new(vec![reward], &self.device).unwrap(),
        //         &tensor_with_goal::<Env>(&next_state_obs, self.plan.last().unwrap(), &self.device).unwrap(),
        //         &Tensor::new(vec![terminated as u8], &self.device).unwrap(),
        //         truncated,
        //     );
        // } else {
        // }
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

