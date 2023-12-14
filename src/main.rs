use {
    anyhow::Result,
    graph_rl::{
        agents::{
            Algorithm,
            DDPG,
            DDPG_SGM,
            configs::{
                DDPG_Config,
                DDPG_SGM_Config,
                AlgorithmConfig,
                OffPolicyConfig,
            },
        },
        envs::{
            Environment,

            PendulumEnv,
            PendulumConfig,

            PointEnv,
            PointEnvConfig,
            PointReward,

            PointMazeEnv,
            PointMazeConfig,
        },
        engines::{
            setup_logging,
            run_experiment_off_policy,
            training_loop_off_policy,
            OffPolicyGUI,
            SgmGUI,
        }
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
    std::fmt::Debug,
    tracing::Level,
};

#[allow(non_camel_case_types)]
#[derive(ValueEnum, Debug, Clone)]
pub enum Alg {
    DDPG,
    DDPG_SGM,
}
impl Alg {
    pub fn name(&self) -> &str {
        match self {
            Alg::DDPG => "ddpg",
            Alg::DDPG_SGM => "ddpg_sgm",
        }
    }
}

#[derive(ValueEnum, Debug, Clone)]
pub enum Env {
    Pendulum,
    Pointenv,
    Pointmaze,
}
impl Env {
    pub fn name(&self) -> &str {
        match self {
            Env::Pendulum => "pendulum",
            Env::Pointenv => "pointenv",
            Env::Pointmaze => "pointmaze",
        }
    }
}

#[derive(ValueEnum, Debug, Clone)]
pub enum Loglevel {
    Error, // put these only during active debugging and then downgrade later
    Warn,  // main events in the program
    Info,  // all the little details
    None,  // don't log anything
}
impl Loglevel {
    pub fn level(&self) -> Option<Level> {
        match self {
            Loglevel::Error => Some(Level::ERROR),
            Loglevel::Warn => Some(Level::WARN),
            Loglevel::Info => Some(Level::INFO),
            Loglevel::None => None,
        }
    }
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,

    /// Setup logging
    #[arg(long, value_enum, default_value_t=Loglevel::None)]
    pub log: Loglevel,

    /// The algorithm to run.
    #[arg(long, value_enum)]
    pub alg: Alg,

    /// The environment to run on.
    #[arg(long, value_enum)]
    pub env: Env,

    /// Run as a GUI instead of just training.
    #[arg(long)]
    pub gui: bool,

    /// Experiment name to use for logging / collecting data.
    #[arg(long)]
    pub name: Option<String>,

    /// Number of runs to collect data on.
    #[arg(long, default_value_t=10)]
    pub runs: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let experiment_name = if let Some(name) = args.name.clone() {
        name
    } else {
        args.env.name().to_owned()
    };

    if let Some(level) = args.log.level() {
        setup_logging(
            &experiment_name,
            Some(level),
        )?;
    }

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::Cuda(CudaDevice::new(0)?)
    };

    match args.env {
        Env::Pendulum => {
            match args.alg {
                Alg::DDPG => {
                    let alg_config = DDPG_Config::pointenv();
                    let env_config = PendulumConfig::default();

                    if args.gui {
                        OffPolicyGUI::<DDPG, PendulumEnv, _, _>::open(
                            env_config,
                            alg_config,
                            device,
                        );
                    } else {
                        run_experiment_off_policy::<DDPG, PendulumEnv, _, _>(
                            &experiment_name,
                            args.runs,
                            env_config,
                            alg_config,
                            &device,
                        )?;
                    }
                },
                Alg::DDPG_SGM => panic!("DDPG_SGM not implemented for Pendulum"),
            }
        }

        Env::Pointenv => {
            let env_config = PointEnvConfig::new(
                // 10.0,
                // 10.0,
                // Some(vec![
                //     ((0.0, 5.0), (5.0, 5.0)).into(),
                //     ((5.0, 5.0), (5.0, 4.0)).into(),
                // ]),
                5.0,
                5.0,
                None,
                30,
                1.0,
                0.5,
                0.1,
                PointReward::Distance,
                42,
            );
            match args.alg {
                Alg::DDPG => {
                    let mut alg_config = DDPG_Config::pointenv();
                    alg_config.set_max_episodes(200);
                    alg_config.set_replay_buffer_capacity(
                        (env_config.width * 10.0 * env_config.height * 10.0) as usize,
                    );
                    if args.gui {
                        OffPolicyGUI::<DDPG, PointEnv, _, _>::open(
                            env_config,
                            alg_config,
                            device,
                        );
                    } else {
                        run_experiment_off_policy::<DDPG, PointEnv, _, _>(
                            &experiment_name,
                            args.runs,
                            env_config,
                            alg_config,
                            &device,
                        )?;
                    }
                },
                Alg::DDPG_SGM => {
                    let mut alg_config = DDPG_SGM_Config::pointenv();
                    alg_config.set_max_episodes(200);
                    alg_config.set_replay_buffer_capacity(
                        (env_config.width * 10.0 * env_config.height * 10.0) as usize,
                    );
                    if args.gui {
                        // let mut env = *PointEnv::new(small_env_config.clone())?;
                        // let mut agent = *DDPG_SGM::from_config(
                        //     &device,
                        //     &alg_config,
                        //     env.observation_space().iter().product::<usize>(),
                        //     env.action_space().iter().product::<usize>(),
                        // )?;
                        // let _ = training_loop_off_policy(
                        //     &mut env,
                        //     &mut agent,
                        //     alg_config.clone(),
                        //     &device,
                        // )?;
                        // agent.new_buffer(alg_config.replay_buffer_capacity() * 100);
                        SgmGUI::<DDPG_SGM<PointEnv>, PointEnv, _, _>::open(
                            env_config,
                            alg_config,
                            None,
                            // Some(agent),
                            device,
                        );
                    } else {
                        run_experiment_off_policy::<DDPG_SGM<PointEnv>, PointEnv, _, _>(
                            &experiment_name,
                            args.runs,
                            env_config,
                            alg_config,
                            &device,
                        )?;
                    }
                },
            }
        }

        Env::Pointmaze => {
            match args.alg {
                Alg::DDPG => {
                    let alg_config = DDPG_Config::pointmaze();
                    let env_config = PointMazeConfig::default();

                    if args.gui {
                        OffPolicyGUI::<DDPG, PointMazeEnv, _, _>::open(
                            env_config,
                            alg_config,
                            device,
                        );
                    } else {
                        run_experiment_off_policy::<DDPG, PointMazeEnv, _, _>(
                            &experiment_name,
                            args.runs,
                            env_config,
                            alg_config,
                            &device,
                        )?;
                    }
                },
                Alg::DDPG_SGM => panic!("DDPG_SGM not implemented for Pointmaze"),
            }
        }
    }
    Ok(())
}
