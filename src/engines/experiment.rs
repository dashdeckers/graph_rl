use {
    super::{
        super::util::write_config,
        run::loop_off_policy,
        ParamAlg,
        ParamEnv,
        ParamRunMode,
    },
    crate::{
        agents::{
            Algorithm,
            OffPolicyAlgorithm,
            SaveableAlgorithm,
        },
        envs::{
            Environment,
            Sampleable,
            TensorConvertible,
        },
        configs::TrainConfig,
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
    },
    tracing::warn,
};

/// Run an experiment with an off-policy algorithm.
///
/// # Arguments
///
/// * `path` - The path to the directory where the collected data will be stored.
/// * `n_repetitions` - The number of repeated, identical runs to perform.
/// * `env_config` - The configuration for the environment.
/// * `alg_config` - The configuration for the algorithm.
/// * `device` - The device to run the experiment on.
#[allow(clippy::too_many_arguments)]
pub fn run_experiment_off_policy<Alg, Env, Obs, Act>(
    path: &dyn AsRef<Path>,
    n_repetitions: usize,
    init_env: ParamEnv<Env, Obs, Act>,
    init_alg: ParamAlg<Alg>,
    train_config: TrainConfig,
    // pretrain_config: Option<TrainConfig>,
    load_model: Option<(String, String)>,
    device: &Device,
) -> Result<()>
where
    Env: Clone + Environment<Action = Act, Observation = Obs>,
    Env::Config: Clone + Serialize,
    Alg: Clone + Algorithm + OffPolicyAlgorithm + SaveableAlgorithm,
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

    let alg_config = match &init_alg {
        ParamAlg::AsAlgorithm(alg) => alg.config().clone(),
        ParamAlg::AsConfig(config) => config.clone(),
    };
    let env_config = match &init_env {
        ParamEnv::AsEnvironment(env) => env.config().clone(),
        ParamEnv::AsConfig(config) => config.clone(),
    };

    create_dir_all(path.as_path())?;
    write_config(&alg_config, path.join("config_algorithm.ron"))?;
    write_config(&env_config, path.join("config_environment.ron"))?;
    write_config(&train_config, path.join("config_training.ron"))?;

    for n in 0..n_repetitions {
        warn!("Collecting data, run {n}/{n_repetitions}");

        // Create the Agent and the Environment

        let mut env = *Env::new(env_config.clone()).unwrap();
        let mut alg = *Alg::from_config(
            device,
            &alg_config,
            env.observation_space().iter().product::<usize>(),
            env.action_space().iter().product::<usize>(),
        ).unwrap();

        // Maybe load model weights

        if let Some((model_path, model_name)) = load_model.clone() {
            warn!("Loading model weights from {model_path} with name {model_name}");
            alg.load(
                &Path::new(&model_path),
                &model_name,
            )?;
        }

        // Pretrain with maxdist (--> different env config!)

        // // Maybe pretrain the Agent

        // if let Some(pretrain_config) = pretrain_config.clone() {
        //     write_config(&pretrain_config, path.join("config_pretraining.ron"))?;

        //     let (pretrain_mc_returns, _) = loop_off_policy(
        //         &mut env,
        //         &mut alg,
        //         ParamRunMode::Train(pretrain_config),
        //         device,
        //     )?;

        //     warn!(
        //         "Pretrained with Avg return: \n{:#?}",
        //         pretrain_mc_returns.iter().sum::<f64>() / pretrain_mc_returns.len() as f64,
        //     );
        // }

        // Train the Agent on the Environment

        let (mc_returns, successes) = loop_off_policy(
            &mut env,
            &mut alg,
            ParamRunMode::Train(train_config.clone()),
            device,
        )?;

        // Write collected data to file

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