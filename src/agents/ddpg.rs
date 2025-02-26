use {
    super::{
        RunMode,
        Algorithm,
        OffPolicyAlgorithm,
        SaveableAlgorithm,
    },
    crate::{
        configs::DDPG_Config,
        components::{
            OuNoise,
            ReplayBuffer,
        },
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
        linear,
        Linear,
        AdamW,
        Optimizer,
        ParamsAdamW,
        VarBuilder,
        VarMap,
    },
    tracing::info,
    std::path::Path,
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
#[derive(Clone)]
struct Actor<'a> {
    varmap: VarMap,
    vb: VarBuilder<'a>,
    network: Vec<Linear>,
    target_network: Vec<Linear>,
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

        let make_layers = |prefix: &str| {
            let layers = vec![
                linear(
                    dims[0].0,
                    dims[0].1,
                    vb.pp(format!("{prefix}-fc0")),
                )?,
                // Activation::Relu,
                linear(
                    dims[1].0,
                    dims[1].1,
                    vb.pp(format!("{prefix}-fc1")),
                )?,
                // Activation::Relu,
                linear(
                    dims[2].0,
                    dims[2].1,
                    vb.pp(format!("{prefix}-fc2")),
                )?,
                // func(|xs| xs.tanh())
            ];
            Ok::<Vec<Linear>, Error>(layers)
        };

        let network = make_layers("actor")?;
        let target_network = make_layers("target-actor")?;

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
        let mut xs = state.clone();

        xs = self.network[0].forward(&xs)?;
        xs = xs.relu()?;
        xs = self.network[1].forward(&xs)?;
        xs = xs.relu()?;
        xs = self.network[2].forward(&xs)?;
        xs = xs.tanh()?;

        Ok(xs)
    }

    fn target_forward(
        &self,
        state: &Tensor,
    ) -> Result<Tensor> {
        let mut xs = state.clone();

        xs = self.target_network[0].forward(&xs)?;
        xs = xs.relu()?;
        xs = self.target_network[1].forward(&xs)?;
        xs = xs.relu()?;
        xs = self.target_network[2].forward(&xs)?;
        xs = xs.tanh()?;

        Ok(xs)
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
#[derive(Clone)]
struct Critic<'a> {
    varmap: VarMap,
    vb: VarBuilder<'a>,
    network: Vec<Linear>,
    target_network: Vec<Linear>,
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

        let make_layers = |prefix: &str| {
            let layers = vec![
                linear(
                    dims[0].0,
                    dims[0].1,
                    vb.pp(format!("{prefix}-fc0")),
                )?,
                // Activation::Relu,
                linear(
                    dims[1].0,
                    dims[1].1,
                    vb.pp(format!("{prefix}-fc1")),
                )?,
                // Activation::Relu,
                linear(
                    dims[2].0,
                    dims[2].1,
                    vb.pp(format!("{prefix}-fc2")),
                )?,
            ];
            Ok::<Vec<Linear>, Error>(layers)
        };

        let network = make_layers("critic")?;
        let target_network = make_layers("target-critic")?;

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
        let mut xs = Tensor::cat(&[action, state], 1)?;

        xs = self.network[0].forward(&xs)?;
        xs = xs.relu()?;
        xs = self.network[1].forward(&xs)?;
        xs = xs.relu()?;
        xs = self.network[2].forward(&xs)?;

        Ok(xs)
    }

    fn target_forward(
        &self,
        state: &Tensor,
        action: &Tensor,
    ) -> Result<Tensor> {
        let mut xs = Tensor::cat(&[action, state], 1)?;

        xs = self.target_network[0].forward(&xs)?;
        xs = xs.relu()?;
        xs = self.target_network[1].forward(&xs)?;
        xs = xs.relu()?;
        xs = self.target_network[2].forward(&xs)?;

        Ok(xs)
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
#[derive(Clone)]
pub struct DDPG<'a> {
    actor: Actor<'a>,
    actor_optim: AdamW,
    critic: Critic<'a>,
    critic_optim: AdamW,
    gamma: f64,
    tau: f64,
    replay_buffer: ReplayBuffer,
    batch_size: usize,
    ou_noise: OuNoise,

    size_state: usize,
    size_action: usize,
    device: Device,
    config: DDPG_Config,
}

impl DDPG<'_> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        size_state: usize,
        size_action: usize,
        hidden_1_size: usize,
        hidden_2_size: usize,
        actor_lr: f64,
        critic_lr: f64,
        gamma: f64,
        tau: f64,
        buffer_capacity: usize,
        batch_size: usize,
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

        let ou_theta = ou_noise.theta();
        let ou_kappa = ou_noise.kappa();
        let ou_sigma = ou_noise.sigma();

        Ok(Self {
            actor,
            actor_optim,
            critic,
            critic_optim,
            gamma,
            tau,
            replay_buffer: ReplayBuffer::new(buffer_capacity),
            batch_size,
            ou_noise,
            size_state,
            size_action,
            device: device.clone(),
            config: DDPG_Config {
                hidden_1_size,
                hidden_2_size,
                actor_learning_rate: actor_lr,
                critic_learning_rate: critic_lr,
                gamma,
                tau,
                replay_buffer_capacity: buffer_capacity,
                training_batch_size: batch_size,
                ou_theta,
                ou_kappa,
                ou_sigma,
            },
        })
    }

    pub fn actor_forward_item(
        &self,
        state: &Tensor,
    ) -> Result<Tensor> {
        self.actor.forward(&state.detach()?.unsqueeze(0)?)?.squeeze(0)
    }

    pub fn critic_forward_item(
        &self,
        state: &Tensor,
        action: &Tensor,
    ) -> Result<Tensor> {
        self.critic.forward(
            &state.detach()?.unsqueeze(0)?,
            &action.detach()?.unsqueeze(0)?,
        )?.squeeze(0)
    }

    pub fn new_buffer(&mut self, buffer_capacity: usize) {
        self.replay_buffer = ReplayBuffer::new(buffer_capacity);
    }

    pub fn set_buffer_capacity(&mut self, buffer_capacity: usize) {
        self.replay_buffer.set_capacity(buffer_capacity);
    }
}

impl Algorithm for DDPG<'_> {
    type Config = DDPG_Config;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn override_config(
        &mut self,
        config: &Self::Config,
    ) {
        self.gamma = config.gamma;
        self.tau = config.tau;
        self.set_buffer_capacity(config.replay_buffer_capacity);
        self.batch_size = config.training_batch_size;

        self.config.gamma = config.gamma;
        self.config.tau = config.tau;
        self.config.replay_buffer_capacity = config.replay_buffer_capacity;
        self.config.training_batch_size = config.training_batch_size;

        if let Ok(noise) = OuNoise::new(
            config.ou_theta,
            config.ou_kappa,
            config.ou_sigma,
            self.size_action,
            &self.device,
        ) {
            self.ou_noise = noise;
            self.config.ou_theta = config.ou_theta;
            self.config.ou_kappa = config.ou_kappa;
            self.config.ou_sigma = config.ou_sigma;
        }
    }

    fn from_config(
        device: &Device,
        config: &DDPG_Config,
        size_state: usize,
        size_action: usize,
    ) -> Result<Box<Self>> {
        Ok(Box::new(Self::new(
            device,
            size_state,
            size_action,
            config.hidden_1_size,
            config.hidden_2_size,
            config.actor_learning_rate,
            config.critic_learning_rate,
            config.gamma,
            config.tau,
            config.replay_buffer_capacity,
            config.training_batch_size,
            OuNoise::new(
                config.ou_theta,
                config.ou_kappa,
                config.ou_sigma,
                size_action,
                device,
            )?,
        )?))
    }

    fn actions(
        &mut self,
        state: &Tensor,
        mode: RunMode,
    ) -> Result<Tensor> {
        // Candle assumes a batch dimension, so when we don't have one we need
        // to pretend we do by un- and resqueezing the state tensor.
        let actions = self.actor.forward(&state.detach()?.unsqueeze(0)?)?.squeeze(0)?;
        Ok(if let RunMode::Train = mode {
            (actions + self.ou_noise.sample()?)?
        } else {
            actions
        })
    }

    fn train(&mut self) -> Result<()> {
        let (states, actions, rewards, next_states, terminated, _) =
            match self.replay_buffer.random_batch(self.batch_size)? {
                Some(v) => v,
                _ => return Ok(()),
            };

        let not_done = (Tensor::ones_like(&terminated) - terminated)?.to_dtype(DType::F64)?;

        let q_target = self.critic.target_forward(&next_states, &self.actor.target_forward(&next_states)?)?;
        let q_target = (rewards + ((not_done * self.gamma)? * q_target)?.detach())?;
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
}


impl OffPolicyAlgorithm for DDPG<'_> {
    fn remember(
        &mut self,
        state: &Tensor,
        action: &Tensor,
        reward: &Tensor,
        next_state: &Tensor,
        terminated: &Tensor,
        truncated: &Tensor,
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

    fn replay_buffer(&self) -> &ReplayBuffer {
        &self.replay_buffer
    }
}

impl SaveableAlgorithm for DDPG<'_> {
    fn save<P: AsRef<Path> + ?Sized>(
        &self,
        path: &P,
        name: &str,
    ) -> Result<()> {
        self.actor.varmap.save(path.as_ref().join(format!("{}-actor.safetensor", name)))?;
        self.critic.varmap.save(path.as_ref().join(format!("{}-critic.safetensor", name)))?;

        Ok(())
    }

    fn load<P: AsRef<Path> + ?Sized>(
        &mut self,
        path: &P,
        name: &str,
    ) -> Result<()> {
        self.actor.varmap.load(path.as_ref().join(format!("{}-actor.safetensor", name)))?;
        self.critic.varmap.load(path.as_ref().join(format!("{}-critic.safetensor", name)))?;

        Ok(())
    }
}