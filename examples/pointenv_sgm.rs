use {
    graph_rl::{
        util::read_config,
        agents::DDPG_SGM,
        envs::{
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
    tracing::Level,
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
    #[arg(long, num_args = 2)]
    pub load_model: Option<Vec<String>>,

    /// Setup logging
    #[arg(long, value_enum, default_value_t=ArgLoglevel::Warn)]
    pub log: ArgLoglevel,

    /// Experiment name to use for logging / collecting data.
    #[arg(long)]
    pub name: String,

    /// Run as a GUI instead of just training.
    #[arg(long, default_value_t=false)]
    pub gui: bool,

    /// Number of repetitions for the experiment.
    #[arg(long, default_value_t=10)]
    pub n_repetitions: usize,
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


    if args.gui {
        //// Run Algorithm in GUI ////

        SgmGUI::<DDPG_SGM<PointEnv>, PointEnv, _, _>::open(
            ParamEnv::AsConfig(match args.env_config {
                Some(env_config) => read_config(env_config)?,
                None => PointEnvConfig::default(),
            }),
            ParamAlg::AsConfig(match args.alg_config {
                Some(alg_config) => read_config(alg_config)?,
                None => DDPG_SGM_Config::default(),
            }),
            ParamRunMode::Train(match args.train_config {
                Some(train_config) => read_config(train_config)?,
                None => TrainConfig::default(),
            }),
            match args.load_model.as_deref() {
                Some([model_path, model_name]) => Some((model_path.to_string(), model_name.to_string())),
                _ => None,
            },
            device,
            1.2,
        );
    } else {
        //// Run Algorithm as Experiment ////

        run_experiment_off_policy::<DDPG_SGM<PointEnv>, PointEnv, _, _>(
            &args.name,
            args.n_repetitions,
            ParamEnv::AsConfig(match args.env_config {
                Some(env_config) => read_config(env_config)?,
                None => PointEnvConfig::default(),
            }),
            ParamAlg::AsConfig(match args.alg_config {
                Some(alg_config) => read_config(alg_config)?,
                None => DDPG_SGM_Config::default(),
            }),
            match args.train_config {
                Some(train_config) => read_config(train_config)?,
                None => TrainConfig::default(),
            },
            match args.load_model.as_deref() {
                Some([model_path, model_name]) => Some((model_path.to_string(), model_name.to_string())),
                _ => None,
            },
            &device,
        )?;
    }

    Ok(())
}