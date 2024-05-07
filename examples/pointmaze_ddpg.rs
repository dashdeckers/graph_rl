use {
    graph_rl::{
        util::read_config,
        cli::{
            ArgLoglevel,
            ArgDevice,
            Args,
        },
        agents::DDPG,
        envs::{
            PointMazeEnv,
            PointMazeConfig,
        },
        configs::{
            DDPG_Config,
            TrainConfig,
        },
        engines::{
            setup_logging,
            run_experiment_off_policy,
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
    clap::Parser,
    anyhow::Result,
    tracing::Level,
};


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

        OffPolicyGUI::<DDPG, PointMazeEnv, _, _>::open(
            ParamEnv::AsConfig(match args.env_config {
                Some(env_config) => read_config(env_config)?,
                None => PointMazeConfig::default(),
            }),
            ParamAlg::AsConfig(match args.alg_config {
                Some(alg_config) => read_config(alg_config)?,
                None => DDPG_Config::default(),
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

        run_experiment_off_policy::<DDPG, PointMazeEnv, _, _>(
            &args.name,
            args.n_repetitions,
            ParamEnv::AsConfig(match args.env_config {
                Some(env_config) => read_config(env_config)?,
                None => PointMazeConfig::default(),
            }),
            ParamAlg::AsConfig(match args.alg_config {
                Some(alg_config) => read_config(alg_config)?,
                None => DDPG_Config::default(),
            }),
            match args.train_config {
                Some(train_config) => read_config(train_config)?,
                None => TrainConfig::default(),
            },
            match args.pretrain_train_config {
                Some(pretrain_train_config) => Some(read_config(pretrain_train_config)?),
                None => None,
            },
            match args.pretrain_env_config {
                Some(pretrain_env_config) => Some(read_config(pretrain_env_config)?),
                None => None,
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