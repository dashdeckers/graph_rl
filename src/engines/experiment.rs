use {
    super::train::training_loop_off_policy,
    crate::{
        agents::{
            Algorithm,
            OffPolicyAlgorithm,
            configs::AlgorithmConfig,
        },
        envs::{
            Environment,
            Sampleable,
            TensorConvertible,
        },
    },
    anyhow::{
        anyhow,
        Result,
    },
    candle_core::Device,
    polars::prelude::{
        DataFrame,
        Series,
        NamedFrom,
        ParquetWriter,
    },
    serde::Serialize,
    std::{
        path::Path,
        fs::{File, create_dir_all},
        io::Write,
    },
    tracing::warn,
};

/// Run an experiment with an off-policy algorithm.
///
/// # Arguments
///
/// * `path` - The path to the directory where the collected data will be stored.
/// * `n_runs` - The number of repeated, identical runs to perform.
/// * `env_config` - The configuration for the environment.
/// * `alg_config` - The configuration for the algorithm.
/// * `device` - The device to run the experiment on.
pub fn run_experiment_off_policy<Alg, Env, Obs, Act>(
    path: &dyn AsRef<Path>,
    n_runs: usize,
    env_config: Env::Config,
    alg_config: Alg::Config,
    device: &Device,
) -> Result<()>
where
    Env: Environment<Action = Act, Observation = Obs>,
    Env::Config: Clone + Serialize,
    Alg: Algorithm + OffPolicyAlgorithm,
    Alg::Config: Clone + Serialize + AlgorithmConfig,
    Obs: Clone + TensorConvertible,
    Act: Clone + TensorConvertible + Sampleable,
{
    let path = Path::new("data/").join(path);

    let alg_config_exists = path.join("config_algorithm.ron").try_exists()?;
    let env_config_exists = path.join("config_environment.ron").try_exists()?;
    if alg_config_exists || env_config_exists {
        Err(anyhow!(concat!(
            "Config files already exist in this directory!\n",
            "I am assuming I would be overwriting existing data!",
        )))?
    }

    create_dir_all(path.as_path())?;

    File::create(path.join("config_algorithm.ron"))?.write_all(
        ron::ser::to_string_pretty(
            &alg_config,
            ron::ser::PrettyConfig::default(),
        )?.as_bytes()
    )?;

    File::create(path.join("config_environment.ron"))?.write_all(
        ron::ser::to_string_pretty(
            &env_config,
            ron::ser::PrettyConfig::default(),
        )?.as_bytes()
    )?;

    for n in 0..n_runs {
        warn!("Collecting data, run {n}/{n_runs}");
        let mut env = *Env::new(env_config.clone())?;
        let mut agent = *Alg::from_config(
            device,
            &alg_config,
            env.observation_space().iter().product::<usize>(),
            env.action_space().iter().product::<usize>(),
        )?;
        let (mc_returns, successes) = training_loop_off_policy(
            &mut env,
            &mut agent,
            alg_config.clone(),
            device,
        )?;

        let mut df = DataFrame::new(vec![
            Series::new(
                &format!("run_{n}_total_rewards"),
                &mc_returns,
            ),
            Series::new(
                &format!("run_{n}_successes"),
                &successes,
            )
        ])?;

        ParquetWriter::new(
            File::create(path.join(format!("run_{n}_data.parquet")))?
        ).finish(&mut df)?;
    }
    Ok(())
}