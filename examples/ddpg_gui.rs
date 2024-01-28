use {
    graph_rl::{
        agents::{
            Algorithm,
            DDPG,
        },
        envs::{
            Environment,
            PointEnv,
            PointEnvConfig,
            PointEnvWalls,
            PointReward,
        },
        configs::{
            DDPG_Config,
            TrainConfig,
        },
        engines::{
            setup_logging,
            run_experiment_off_policy,
            loop_off_policy,
            ParamRunMode,
            ParamEnv,
            ParamAlg,
            OffPolicyGUI,
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

    /// Number of pretraining runs to perform.
    #[arg(long, default_value = "0")]
    pub pretrain: usize,

    /// Setup logging
    #[arg(long, value_enum, default_value_t=ArgLoglevel::Warn)]
    pub log: ArgLoglevel,

    /// Experiment name to use for logging / collecting data.
    #[arg(long, default_value = "ddpg-test")]
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


    //// Create the PointEnv Environment for Training ////

    let mut pointenv = *PointEnv::new(
        PointEnvConfig::new(
            10.0,
            10.0,
            PointEnvWalls::None,
            10,
            1.0,
            0.5,
            Some(2.5),
            0.1,
            PointReward::Distance,
            42,
        ),
    )?;


    //// Create DDPG Algorithm ////

    let ddpg = *DDPG::from_config(
        &device,
        &DDPG_Config::small(),
        pointenv.observation_space().iter().product::<usize>(),
        pointenv.action_space().iter().product::<usize>(),
    )?;


    //// Create the TrainConfig ////

    let train_config = TrainConfig::new(
        300,
        30,
        500,
    );


    //// Pretrain DDPG_SGM Algorithm ////

    for n in 0..args.pretrain {
        let (mc_returns, successes) = loop_off_policy(
            &mut pointenv,
            &mut ddpg.clone(),
            ParamRunMode::Train(train_config.clone()),
            &device,
        )?;

        warn!(
            "Pretrain run #{} - Avg return: {:.3}, Successes: {}/{}",
            n,
            mc_returns.iter().sum::<f64>() / mc_returns.len() as f64,
            successes.iter().filter(|&&s| s).count(),
            successes.len(),
        );
    }


    if args.gui {
        //// Check Pretrained DDPG Performance via GUI ////

        OffPolicyGUI::<DDPG, PointEnv, _, _>::open(
            ParamEnv::AsEnvironment(pointenv),
            ParamAlg::AsAlgorithm(ddpg),
            ParamRunMode::Train(train_config),
            device,
        );
    } else {
        //// Run Pretrained DDPG Algorithm in Experiment ////

        run_experiment_off_policy::<DDPG, PointEnv, _, _>(
            &args.name,
            100,
            ParamEnv::AsEnvironment(pointenv),
            ParamAlg::AsAlgorithm(ddpg),
            ParamRunMode::Train(train_config),
            &device,
        )?;
    }

    Ok(())
}