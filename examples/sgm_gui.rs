use {
    graph_rl::{
        agents::{
            Algorithm,
            DDPG_SGM,
        },
        envs::{
            Environment,
            PointEnv,
            PointEnvConfig,
            PointEnvWalls,
            PointReward,
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

    /// Setup logging
    #[arg(long, value_enum, default_value_t=ArgLoglevel::Warn)]
    pub log: ArgLoglevel,

    /// Experiment name to use for logging / collecting data.
    #[arg(long, default_value = "sgm-test")]
    pub name: String,

    /// Run as a GUI instead of just training.
    #[arg(long, default_value_t=true)]
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


    //// Create the PointEnv Environment for Training ////

    let pointenv = *PointEnv::new(
        PointEnvConfig::new(
            10.0,
            10.0,
            PointEnvWalls::None,
            30,
            1.0,
            0.5,
            None,
            0.1,
            PointReward::Distance,
            42,
        ),
    )?;


    //// Create DDPG_SGM Algorithm ////

    let ddpg_sgm = *DDPG_SGM::from_config(
        &device,
        &DDPG_SGM_Config::pointenv(),
        pointenv.observation_space().iter().product::<usize>(),
        pointenv.action_space().iter().product::<usize>(),
    )?;


    //// Create the TrainConfig ////

    let train_config = TrainConfig::new(
        200,
        30,
        500,
    );


    if args.gui {
        //// Check Pretrained DDPG_SGM Performance via GUI ////

        SgmGUI::<DDPG_SGM<PointEnv>, PointEnv, _, _>::open(
            ParamEnv::AsEnvironment(pointenv),
            ParamAlg::AsAlgorithm(ddpg_sgm),
            ParamRunMode::Train(train_config),
            device,
        );
    } else {
        //// Run Pretrained DDPG_SGM Algorithm in Experiment ////

        run_experiment_off_policy::<DDPG_SGM<PointEnv>, PointEnv, _, _>(
            &args.name,
            100,
            ParamEnv::AsEnvironment(pointenv),
            ParamAlg::AsAlgorithm(ddpg_sgm),
            ParamRunMode::Train(train_config),
            &device,
        )?;
    }

    Ok(())
}