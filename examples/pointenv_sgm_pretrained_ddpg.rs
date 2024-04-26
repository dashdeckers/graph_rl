use {
    graph_rl::{
        util::read_config,
        agents::{
            Algorithm,
            SaveableAlgorithm,
            DDPG_SGM,
            DDPG,
        },
        envs::{
            Environment,
            PointEnv,
            PointEnvConfig,
        },
        configs::{
            DDPG_SGM_Config,
            TrainConfig,
        },
        engines::{
            setup_logging,
            run_experiment_off_policy,
            loop_off_policy,
            ParamRunMode,
            ParamEnv,
            ParamAlg,
            SgmGUI,
        },
    },
    candle_core::{
        Device,
        CudaDevice,
        backend::BackendDevice,
    },
    clap::{
        Parser,
        ValueEnum,
    },
    anyhow::Result,
    tracing::{
        Level,
        warn,
    },
    std::path::Path,
};


#[derive(ValueEnum, Debug, Clone)]
enum ArgLoglevel {
    Error,
    Warn,
    Info,
    None,
}

#[derive(ValueEnum, Debug, Clone)]
enum ArgDevice {
    Cpu,
    Cuda,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Device to run on (e.g. CPU / GPU).
    #[arg(long, value_enum, default_value_t=ArgDevice::Cpu)]
    pub device: ArgDevice,

    /// Pretrain the model according to the given config.
    #[arg(long)]
    pub pretrain_config: Option<String>,

    /// Train the model according to the given config.
    #[arg(long)]
    pub train_config: Option<String>,

    /// Environment config.
    #[arg(long)]
    pub env_config: Option<String>,

    /// Algorithm config.
    #[arg(long)]
    pub alg_config: Option<String>,

    /// Load a pretrained model from a file.
    #[arg(long)]
    pub load_model: Option<String>,

    /// Setup logging
    #[arg(long, value_enum, default_value_t=ArgLoglevel::Warn)]
    pub log: ArgLoglevel,

    /// Experiment name to use for logging / collecting data.
    #[arg(long)]
    pub name: String,

    /// Run as a GUI instead of just training.
    #[arg(long, default_value_t=false)]
    pub gui: bool,
}


fn main() -> Result<()> {
    let args = Args::parse();

    setup_logging(
        &args.name,
        match args.log {
            ArgLoglevel::Error => Some(Level::ERROR),
            ArgLoglevel::Warn => Some(Level::WARN),
            ArgLoglevel::Info => Some(Level::INFO),
            ArgLoglevel::None => None,
        },
    )?;

    let device = match args.device {
        ArgDevice::Cpu => Device::Cpu,
        ArgDevice::Cuda => Device::Cuda(CudaDevice::new(0)?),
    };


    //// Create the Environment ////

    let mut env = *PointEnv::new(match args.env_config {
        Some(config_path) => read_config(config_path)?,
        None => PointEnvConfig::default(),
    })?;


    //// Read the Algorithm Config ////

    let alg_config = match args.alg_config {
        Some(config_path) => read_config(config_path)?,
        None => DDPG_SGM_Config::default(),
    };


    //// Create DDPG ////

    let mut ddpg = *DDPG::from_config(
        &device,
        &alg_config.ddpg,
        env.observation_space().iter().product::<usize>(),
        env.action_space().iter().product::<usize>(),
    )?;


    //// Maybe load DDPG Weights ////

    if let Some(model_path) = args.load_model {
        ddpg.load(
            &Path::new(&model_path),
            &args.name,
        )?;
    }


    //// Maybe Pretrain DDPG ////

    if let Some(config_path) = args.pretrain_config {
        let (mc_returns, _) = loop_off_policy(
            &mut env,
            &mut ddpg,
            ParamRunMode::Train(read_config(config_path)?),
            &device,
        )?;

        ddpg.save(
            &Path::new("data/").join(&args.name),
            &format!("{}-pretrained", &args.name),
        )?;

        warn!(
            "Pretrained with: \n{:#?}",
            mc_returns,
        );
    }


    //// Create the TrainConfig ////

    let train_config = match args.train_config {
        Some(config_path) => read_config(config_path)?,
        None => TrainConfig::default(),
    };


    //// Create DDPG_SGM ////

    let ddpg_sgm = *DDPG_SGM::from_config_with_ddpg(
        &device,
        &alg_config,
        ddpg,
    )?;


    if args.gui {
        //// Check Pretrained DDPG_SGM via GUI ////

        SgmGUI::<DDPG_SGM<PointEnv>, PointEnv, _, _>::open(
            ParamEnv::AsEnvironment(env),
            ParamAlg::AsAlgorithm(ddpg_sgm),
            ParamRunMode::Train(train_config),
            device,
        );
    } else {
        //// Run Pretrained DDPG_SGM in Experiment ////

        run_experiment_off_policy::<DDPG_SGM<PointEnv>, PointEnv, _, _>(
            &args.name,
            10,
            ParamEnv::AsEnvironment(env),
            ParamAlg::AsAlgorithm(ddpg_sgm),
            ParamRunMode::Train(train_config),
            &device,
        )?;
    }

    Ok(())
}