use {
    super::{
        run::loop_off_policy,
        ParamAlg,
        ParamEnv,
        ParamRunMode,
    },
    crate::{
        agents::{
            Algorithm,
            OffPolicyAlgorithm,
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
    serde::Serialize,
    candle_core::Device,
    polars::prelude::{
        DataFrame,
        Series,
        NamedFrom,
        ParquetWriter,
    },
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
    env: ParamEnv<Env, Obs, Act>,
    alg: ParamAlg<Alg>,
    run_mode: ParamRunMode,
    device: &Device,
) -> Result<()>
where
    Env: Environment<Action = Act, Observation = Obs>,
    Env::Config: Clone + Serialize,
    Alg: Clone + Algorithm + OffPolicyAlgorithm,
    Alg::Config: Clone + Serialize,
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

    let alg_config = match &alg {
        ParamAlg::AsAlgorithm(alg) => alg.config().clone(),
        ParamAlg::AsConfig(config) => config.clone(),
    };
    let env_config = match &env {
        ParamEnv::AsEnvironment(env) => env.config().clone(),
        ParamEnv::AsConfig(config) => config.clone(),
    };

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

    File::create(path.join("config_training.ron"))?.write_all(
        ron::ser::to_string_pretty(
            &run_mode,
            ron::ser::PrettyConfig::default(),
        )?.as_bytes()
    )?;

    let mut env = match env {
        ParamEnv::AsEnvironment(env) => env,
        ParamEnv::AsConfig(config) => {
            *Env::new(config.clone())?
        },
    };
    let size_state = env.observation_space().iter().product::<usize>();
    let size_action = env.action_space().iter().product::<usize>();

    for n in 0..n_runs {
        warn!("Collecting data, run {n}/{n_runs}");
        let mut alg = match &alg {
            ParamAlg::AsAlgorithm(alg) => alg.clone(),
            ParamAlg::AsConfig(config) => {
                *Alg::from_config(
                    device,
                    &config.clone(),
                    size_state,
                    size_action,
                )?
            },
        };
        let (mc_returns, successes) = loop_off_policy(
            &mut env,
            &mut alg,
            run_mode.clone(),
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