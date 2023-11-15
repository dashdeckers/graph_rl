use {
    crate::{
        components::{
            OuNoise,
            ReplayBuffer,
        },
        engine::RunMode,
        TrainingConfig,
    },
    candle_core::{
        DType,
        Device,
        Error,
        Module,
        Result,
        Tensor,
        Var,
    },
    candle_nn::{
        func,
        linear,
        sequential::seq,
        Activation,
        AdamW,
        Optimizer,
        ParamsAdamW,
        Sequential,
        VarBuilder,
        VarMap,
    },
    tracing::info,
};

fn track(
    varmap: &mut VarMap,
    vb: &VarBuilder,
    target_prefix: &str,
    network_prefix: &str,
    dims: &[(usize, usize)],
    tau: f64,
) -> Result<()> {
    for (i, &(in_dim, out_dim)) in dims.iter().enumerate() {
        let target_w = vb.get((out_dim, in_dim), &format!("{target_prefix}-fc{i}.weight"))?;
        let network_w = vb.get((out_dim, in_dim), &format!("{network_prefix}-fc{i}.weight"))?;
        varmap.set_one(
            format!("{target_prefix}-fc{i}.weight"),
            ((tau * network_w)? + ((1.0 - tau) * target_w)?)?,
        )?;

        let target_b = vb.get(out_dim, &format!("{target_prefix}-fc{i}.bias"))?;
        let network_b = vb.get(out_dim, &format!("{network_prefix}-fc{i}.bias"))?;
        varmap.set_one(
            format!("{target_prefix}-fc{i}.bias"),
            ((tau * network_b)? + ((1.0 - tau) * target_b)?)?,
        )?;
    }
    Ok(())
}

#[allow(dead_code)]
struct Actor<'a> {
    varmap: VarMap,
    vb: VarBuilder<'a>,
    network: Sequential,
    target_network: Sequential,
    dims: Vec<(usize, usize)>,
}

impl Actor<'_> {
    fn new(
        device: &Device,
        dtype: DType,
        dims: &[(usize, usize)],
    ) -> Result<Self> {
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);

        let make_network = |prefix: &str| {
            let seq = seq()
                .add(linear(
                    dims[0].0,
                    dims[0].1,
                    vb.pp(format!("{prefix}-fc0")),
                )?)
                .add(Activation::Relu)
                .add(linear(
                    dims[1].0,
                    dims[1].1,
                    vb.pp(format!("{prefix}-fc1")),
                )?)
                .add(Activation::Relu)
                .add(linear(
                    dims[2].0,
                    dims[2].1,
                    vb.pp(format!("{prefix}-fc2")),
                )?)
                .add(func(|xs| xs.tanh()));
            Ok::<Sequential, Error>(seq)
        };

        let network = make_network("actor")?;
        let target_network = make_network("target-actor")?;

        // this sets the two networks to be equal to each other using tau = 1.0
        track(&mut varmap, &vb, "target-actor", "actor", dims, 1.0)?;

        Ok(Self {
            varmap,
            vb,
            network,
            target_network,
            dims: dims.to_vec(),
        })
    }

    fn forward(
        &self,
        state: &Tensor,
    ) -> Result<Tensor> {
        self.network.forward(state)
    }

    fn target_forward(
        &self,
        state: &Tensor,
    ) -> Result<Tensor> {
        self.target_network.forward(state)
    }

    fn track(
        &mut self,
        tau: f64,
    ) -> Result<()> {
        track(
            &mut self.varmap,
            &self.vb,
            "target-actor",
            "actor",
            &self.dims,
            tau,
        )
    }
}

#[allow(dead_code)]
struct Critic<'a> {
    varmap: VarMap,
    vb: VarBuilder<'a>,
    network: Sequential,
    target_network: Sequential,
    dims: Vec<(usize, usize)>,
}

impl Critic<'_> {
    fn new(
        device: &Device,
        dtype: DType,
        dims: &[(usize, usize)],
    ) -> Result<Self> {
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);

        let make_network = |prefix: &str| {
            let seq = seq()
                .add(linear(
                    dims[0].0,
                    dims[0].1,
                    vb.pp(format!("{prefix}-fc0")),
                )?)
                .add(Activation::Relu)
                .add(linear(
                    dims[1].0,
                    dims[1].1,
                    vb.pp(format!("{prefix}-fc1")),
                )?)
                .add(Activation::Relu)
                .add(linear(
                    dims[2].0,
                    dims[2].1,
                    vb.pp(format!("{prefix}-fc2")),
                )?);
            Ok::<Sequential, Error>(seq)
        };

        let network = make_network("critic")?;
        let target_network = make_network("target-critic")?;

        // this sets the two networks to be equal to each other using tau = 1.0
        track(&mut varmap, &vb, "target-critic", "critic", dims, 1.0)?;

        Ok(Self {
            varmap,
            vb,
            network,
            target_network,
            dims: dims.to_vec(),
        })
    }

    fn forward(
        &self,
        state: &Tensor,
        action: &Tensor,
    ) -> Result<Tensor> {
        let xs = Tensor::cat(&[action, state], 1)?;
        self.network.forward(&xs)
    }

    fn target_forward(
        &self,
        state: &Tensor,
        action: &Tensor,
    ) -> Result<Tensor> {
        let xs = Tensor::cat(&[action, state], 1)?;
        self.target_network.forward(&xs)
    }

    fn track(
        &mut self,
        tau: f64,
    ) -> Result<()> {
        track(
            &mut self.varmap,
            &self.vb,
            "target-critic",
            "critic",
            &self.dims,
            tau,
        )
    }
}

#[allow(dead_code)]
#[allow(clippy::upper_case_acronyms)]
pub struct DDPG<'a> {
    actor: Actor<'a>,
    actor_optim: AdamW,
    critic: Critic<'a>,
    critic_optim: AdamW,
    gamma: f64,
    tau: f64,
    replay_buffer: ReplayBuffer,
    ou_noise: OuNoise,

    size_state: usize,
    size_action: usize,
    pub run_mode: RunMode,
}

impl DDPG<'_> {
    pub fn from_config(
        device: &Device,
        config: &TrainingConfig,
        size_state: usize,
        size_action: usize,
    ) -> Result<Self> {
        Self::new(
            device,
            size_state,
            size_action,
            config.hidden_1_size,
            config.hidden_2_size,
            RunMode::Train,
            config.actor_learning_rate,
            config.critic_learning_rate,
            config.gamma,
            config.tau,
            config.replay_buffer_capacity,
            OuNoise::new(config.ou_mu, config.ou_theta, config.ou_sigma, size_action)?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        size_state: usize,
        size_action: usize,
        hidden_1_size: usize,
        hidden_2_size: usize,
        run_mode: RunMode,
        actor_lr: f64,
        critic_lr: f64,
        gamma: f64,
        tau: f64,
        buffer_capacity: usize,
        ou_noise: OuNoise,
    ) -> Result<Self> {
        let filter_by_prefix = |varmap: &VarMap, prefix: &str| {
            varmap
                .data()
                .lock()
                .unwrap()
                .iter()
                .filter_map(|(name, var)| name.starts_with(prefix).then_some(var.clone()))
                .collect::<Vec<Var>>()
        };

        let actor = Actor::new(
            device,
            DType::F64,
            &[
                (size_state, hidden_1_size),
                (hidden_1_size, hidden_2_size),
                (hidden_2_size, size_action),
            ],
        )?;
        let actor_optim = AdamW::new(
            filter_by_prefix(&actor.varmap, "actor"),
            ParamsAdamW {
                lr: actor_lr,
                ..Default::default()
            },
        )?;

        let critic = Critic::new(
            device,
            DType::F64,
            &[
                (size_state + size_action, hidden_1_size),
                (hidden_1_size, hidden_2_size),
                (hidden_2_size, 1),
            ],
        )?;
        let critic_optim = AdamW::new(
            filter_by_prefix(&critic.varmap, "critic"),
            ParamsAdamW {
                lr: critic_lr,
                ..Default::default()
            },
        )?;

        Ok(Self {
            actor,
            actor_optim,
            critic,
            critic_optim,
            gamma,
            tau,
            replay_buffer: ReplayBuffer::new(buffer_capacity),
            ou_noise,
            size_state,
            size_action,
            run_mode,
        })
    }

    pub fn remember(
        &mut self,
        state: &Tensor,
        action: &Tensor,
        reward: &Tensor,
        next_state: &Tensor,
        terminated: bool,
        truncated: bool,
    ) {
        info!(
            concat!(
                "\nPushing to replay buffer:",
                "\n{state:?}",
                "\n{action:?}",
                "\n{reward:?}",
                "\n{next_state:?}",
            ),
            state = state,
            action = action,
            reward = reward,
            next_state = next_state,
        );
        self.replay_buffer
            .push(state, action, reward, next_state, terminated, truncated)
    }

    pub fn actions(
        &mut self,
        state: &Tensor,
    ) -> Result<Vec<f64>> {
        let actions = self
            .actor
            .forward(&state.detach()?.unsqueeze(0)?)?
            .squeeze(0)?;

        let actions = if let RunMode::Train = self.run_mode {
            (actions + self.ou_noise.sample()?)?
        } else {
            actions
        };
        actions.to_vec1::<f64>()
    }

    pub fn train(
        &mut self,
        batch_size: usize,
    ) -> Result<()> {
        let (states, actions, rewards, next_states, _, _) =
            match self.replay_buffer.random_batch(batch_size)? {
                Some(v) => v,
                _ => return Ok(()),
            };

        let q_target = self
            .critic
            .target_forward(&next_states, &self.actor.target_forward(&next_states)?)?;
        let q_target = (rewards + (self.gamma * q_target)?.detach())?;
        let q = self.critic.forward(&states, &actions)?;
        let diff = (q_target - q)?;

        let critic_loss = diff.sqr()?.mean_all()?;
        self.critic_optim.backward_step(&critic_loss)?;

        let actor_loss = self
            .critic
            .forward(&states, &self.actor.forward(&states)?)?
            .mean_all()?
            .neg()?;
        self.actor_optim.backward_step(&actor_loss)?;

        self.critic.track(self.tau)?;
        self.actor.track(self.tau)?;

        Ok(())
    }

    pub fn replay_buffer(&self) -> &ReplayBuffer {
        &self.replay_buffer
    }
}
