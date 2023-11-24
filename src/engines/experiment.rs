use {
    super::train::training_loop_off_policy,
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
        fmt::Debug,
        hash::Hash,
    },
    tracing::warn,
};


pub fn run_experiment_off_policy<Alg, Env, Obs, Act>(
    path: &dyn AsRef<Path>,
    n_runs: usize,
    env: &mut Env,
    config: Alg::Config,
    device: &Device,
) -> Result<()>
where
    Env: Environment<Action = Act, Observation = Obs>,
    Env::Config: Clone + Serialize,
    Alg: Algorithm + OffPolicyAlgorithm,
    Alg::Config: Clone + Serialize + AlgorithmConfig + OffPolicyConfig + SgmConfig,
    Obs: Debug + Clone + Eq + Hash + TensorConvertible + DistanceMeasure,
    Act: Clone + TensorConvertible + Sampleable,
{
    let path = Path::new("data/").join(path);

    if path.join("config_algorithm.ron").try_exists()? {
        Err(anyhow!(concat!(
            "Algorithm config already exists in this directory!\n",
            "I am assuming I would be overwriting existing data!",
        )))?
    }

    create_dir_all(path.as_path())?;

    File::create(path.join("config_algorithm.ron"))?.write_all(
        ron::ser::to_string_pretty(
            &config,
            ron::ser::PrettyConfig::default(),
        )?.as_bytes()
    )?;

    File::create(path.join("config_environment.ron"))?.write_all(
        ron::ser::to_string_pretty(
            &env.config(),
            ron::ser::PrettyConfig::default(),
        )?.as_bytes()
    )?;

    for n in 0..n_runs {
        warn!("Collecting data, run {n}/{n_runs}");
        let mut agent = *Alg::from_config(
            device,
            &config,
            env.observation_space().iter().product::<usize>(),
            env.action_space().iter().product::<usize>(),
        )?;
        let (mc_returns, successes) = training_loop_off_policy(
            env,
            &mut agent,
            config.clone(),
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