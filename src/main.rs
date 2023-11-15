use {
    anyhow::Result,
    candle_core::Device,
    clap::{
        Parser,
        ValueEnum,
    },
    graph_rl::{
        agents::{
            DDPG,
            DDPGConfig,
            Algorithm,
        },
        envs::{
            Environment,
            PendulumEnv,

            PointEnv,
            PointEnvConfig,
            PointReward,

            PointMazeEnv,
        },
        gui::GUI,
        engine::run_n,
        logging::setup_logging,
    },
    tracing::Level,
};

#[derive(ValueEnum, Debug, Clone)]
enum Env {
    Pendulum,
    Pointenv,
    Pointmaze,
}

#[derive(ValueEnum, Debug, Clone)]
enum Loglevel {
    Error, // put these only during active debugging and then downgrade later
    Warn,  // main events in the program
    Info,  // all the little details
    None,  // don't log anything
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Setup logging
    #[arg(long, value_enum, default_value_t=Loglevel::None)]
    log: Loglevel,

    /// The environment to run.
    #[arg(long, value_enum)]
    env: Env,

    /// Run as a GUI instead of just training.
    #[arg(long)]
    gui: bool,

    /// File to write the results to.
    #[arg(long)]
    output: Option<String>,
}

// NOW

// >- Distributional RL

// LATER

// >- Add Cuda as a Device and get that working on the server
// >- Put the Candle "cuda" feature behind a cfg() flag
//    `-> https://doc.rust-lang.org/cargo/reference/features.html


fn main() -> Result<()> {
    let args = Args::parse();
    match args.log {
        Loglevel::Error => setup_logging(
            &"debug.log",
            Some(Level::ERROR),
            Some(Level::ERROR),
        )?,
        Loglevel::Warn => setup_logging(
            &"debug.log",
            Some(Level::WARN),
            Some(Level::WARN),
        )?,
        Loglevel::Info => setup_logging(
            &"debug.log",
            Some(Level::INFO),
            Some(Level::INFO),
        )?,
        Loglevel::None => (),
    };

    // let device = &if args.cpu { Device::Cpu } else { Device::Cpu }; // TODO: Cuda
    let device = Device::Cpu;
    match args.env {
        Env::Pendulum => {
            let mut env = *PendulumEnv::new(Default::default())?;
            let config = DDPGConfig::pendulum();

            if args.gui {
                let agent = *DDPG::from_config(
                    &device,
                    &config,
                    env.observation_space().iter().product::<usize>(),
                    env.action_space().iter().product::<usize>(),
                )?;
                GUI::open(
                    env,
                    agent,
                    config,
                    device,
                );
            } else {
                run_n::<DDPG, _, _, _>(
                    &"pendulum_data.parquet",
                    &"pendulum_data.ron",
                    10,
                    &mut env,
                    config,
                    &device,
                )?;
            }
        }

        Env::Pointenv => {
            let mut env = *PointEnv::new(PointEnvConfig::new(
                5,
                5,
                None,
                30,
                1.0,
                0.5,
                0.1,
                PointReward::Distance,
                42,
            ))?;
            let config = DDPGConfig::pointenv();

            if args.gui {
                let agent = *DDPG::from_config(
                    &device,
                    &config,
                    env.observation_space().iter().product::<usize>(),
                    env.action_space().iter().product::<usize>(),
                )?;
                GUI::open(env, agent, config, device);
            } else {
                run_n::<DDPG, _, _, _>(
                    &"pointenv_data.parquet",
                    &"pointenv_data.ron",
                    10,
                    &mut env,
                    config,
                    &device,
                )?;
            }
        }

        Env::Pointmaze => {
            let mut env = *PointMazeEnv::new(Default::default())?;
            let config = DDPGConfig::pointmaze();

            if args.gui {
                let agent = *DDPG::from_config(
                    &device,
                    &config,
                    env.observation_space().iter().product::<usize>(),
                    env.action_space().iter().product::<usize>(),
                )?;
                GUI::open(env, agent, config, device);
            } else {
                run_n::<DDPG, _, _, _>(
                    &"pointmaze_data.parquet",
                    &"pointmaze_data.ron",
                    10,
                    &mut env,
                    config,
                    &device,
                )?;
            }
        }
    }
    Ok(())
}
