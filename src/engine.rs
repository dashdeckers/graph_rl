use {
    crate::{
        agents::{
            Algorithm,
            OffPolicyAlgorithm,
            AlgorithmConfig,
            OffPolicyConfig,
            SgmConfig,
        },
        envs::{
            DistanceMeasure,
            Environment,
            Sampleable,
            TensorConvertible,
            VectorConvertible,
        },
        RunMode,
    },
    anyhow::Result,
    candle_core::{
        Device,
        Tensor,
    },
    polars::prelude::{
        DataFrame,
        Series,
        NamedFrom,
        ParquetWriter,
    },
    serde::Serialize,
    rand::{
        thread_rng,
        Rng,
    },
    std::{
        path::Path,
        fs::File,
        io::Write,
        fmt::Debug,
        hash::Hash,
    },
    tracing::warn,
};


pub fn run_n<Alg, Env, Obs, Act>(
    data_path: &dyn AsRef<Path>,
    config_path: &dyn AsRef<Path>,
    n_runs: usize,
    env: &mut Env,
    config: Alg::Config,
    device: &Device,
) -> Result<DataFrame>
where
    Alg: Algorithm + OffPolicyAlgorithm,
    Alg::Config: AlgorithmConfig + OffPolicyConfig + SgmConfig + Clone + Serialize,
    Env: Environment<Action = Act, Observation = Obs>,
    Obs: Debug + Clone + Eq + Hash + TensorConvertible + DistanceMeasure,
    Act: Clone + VectorConvertible + Sampleable,
{
    let mut cols: Vec<Series> = Vec::new();
    for n in 0..n_runs {
        warn!("Collecting data, run {n}/{n_runs}");
        let mut agent = *Alg::from_config(
            device,
            &config,
            env.observation_space().iter().product::<usize>(),
            env.action_space().iter().product::<usize>(),
        )?;
        let (mc_returns, successes) = train(
            env,
            &mut agent,
            config.clone(),
            device,
        )?;
        cols.push(Series::new(
            &format!("run_{n}_total_rewards"),
            &mc_returns,
        ));
        cols.push(Series::new(
            &format!("run_{n}_successes"),
            &successes,
        ));
    }

    let mut config_file = File::create(config_path)?;
    config_file.write_all(
        ron::ser::to_string_pretty(
            &config,
            ron::ser::PrettyConfig::default(),
        )?.as_bytes()
    )?;

    let data_file = File::create(data_path)?;
    let mut df = DataFrame::new(cols)?;
    ParquetWriter::new(data_file).finish(&mut df)?;
    Ok(df)
}

pub fn train<Alg, Env, Obs, Act>(
    env: &mut Env,
    agent: &mut Alg,
    config: Alg::Config,
    device: &Device,
) -> Result<(Vec<f64>, Vec<bool>)>
where
    Alg: Algorithm + OffPolicyAlgorithm,
    Alg::Config: AlgorithmConfig + OffPolicyConfig + SgmConfig,
    Env: Environment<Action = Act, Observation = Obs>,
    Obs: Debug + Clone + Eq + Hash + TensorConvertible + DistanceMeasure,
    Act: Clone + VectorConvertible + Sampleable,
{
    warn!("action space: {:?}", env.action_space());
    warn!("observation space: {:?}", env.observation_space());

    let mut steps_taken = 0;
    let mut mc_returns = Vec::new();
    let mut successes = Vec::new();
    let mut rng = rand::thread_rng();

    for episode in 0..config.max_episodes() {
        let mut total_reward = 0.0;
        env.reset(rng.gen::<u64>())?;

        loop {
            let observation = env.current_observation();
            let state = &<Obs>::to_tensor(observation, device)?;

            // select an action, or randomly sample one
            let action = if steps_taken < config.initial_random_actions() {
                <Act>::to_vec(<Act>::sample(&mut rng, &env.action_domain()))
            } else {
                agent.actions(state)?
            };

            let step = env.step(<Act>::from_vec(action.clone()))?;
            total_reward += step.reward;
            steps_taken += 1;

            agent.remember(
                state,
                &Tensor::new(action, device)?,
                &Tensor::new(vec![step.reward], device)?,
                &<Obs>::to_tensor(step.observation, device)?,
                step.terminated,
                step.truncated,
            );

            if step.terminated || step.truncated {
                successes.push(step.terminated);
                break;
            }
        }

        warn!("episode {episode} with total reward of {total_reward}");
        mc_returns.push(total_reward);

        if let RunMode::Train = agent.run_mode() {
            for _ in 0..config.training_iterations() {
                agent.train()?;
            }
        }
    }
    env.reset(rng.gen::<u64>())?;
    Ok((mc_returns, successes))
}

pub fn tick<Alg, Env, Obs, Act>(
    env: &mut Env,
    agent: &mut Alg,
    device: &Device,
) -> Result<()>
where
    Alg: Algorithm + OffPolicyAlgorithm,
    Alg::Config: AlgorithmConfig + OffPolicyConfig + SgmConfig,
    Env: Environment<Action = Act, Observation = Obs>,
    Obs: Clone + TensorConvertible,
    Act: Clone + VectorConvertible,
{
    let action = agent.actions(&<Obs>::to_tensor(env.current_observation(), device)?)?;
    let step = if let Ok(step) = env.step(<Act>::from_vec(action.clone())) {
        step
    } else {
        env.reset(thread_rng().gen::<u64>())?;
        env.step(<Act>::from_vec(action))?
    };

    if step.terminated || step.truncated {
        env.reset(thread_rng().gen::<u64>())?;
    }
    Ok(())
}

