extern crate intel_mkl_src;

use candle_core::Device;
use anyhow::Result;
use tracing::Level;
use clap::{Parser, ValueEnum};

use graph_rl::{
    ddpg::DDPG,
    envs::{
        PendulumEnv,
        PointEnv,
        Environment,
    },
    util::setup_logging,
    gui::GUI,
    TrainingConfig,
    RunMode,
    run,
};


#[derive(ValueEnum, Debug, Clone)]
enum Env {
    Pendulum,
    Pointenv,
}

#[derive(ValueEnum, Debug, Clone)]
enum Loglevel {
    Error,  // put these only during active debugging and then downgrade later
    Warn,   // main events in the program
    Info,   // all the little details
    None,   // don't log anything
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
}

// NOW

// TODO: Learning is broken
// Pendulum with reward mode Distance the agent almost avoids learning?
// 




// LATER

// >- Add AntMaze / PointMaze as Goal-Aware Environments
//    `-> these return a dict with 3 keys of Vec<64> instead of just a Vec<f64>

// >- Add Cuda as a Device and get that working on the server
// >- Put the Candle "cuda" feature behind a cfg() flag
//    `-> https://doc.rust-lang.org/cargo/reference/features.html


fn main() -> Result<()> {
    let args = Args::parse();
    match args.log {
        Loglevel::Error => setup_logging(
            "debug.log".into(),
            Some(Level::ERROR),
            Some(Level::ERROR),
        )?,
        Loglevel::Warn => setup_logging(
            "debug.log".into(),
            Some(Level::WARN),
            Some(Level::WARN),
        )?,
        Loglevel::Info => setup_logging(
            "debug.log".into(),
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
            let config = TrainingConfig::pendulum();

            let size_state = env.observation_space().iter().product::<usize>();
            let size_action = env.action_space().iter().product::<usize>();

            let mut agent = DDPG::from_config(&device, &config, size_state, size_action)?;

            if args.gui {
                GUI::open(env, agent, config, device);
            } else {
                run(
                    &mut env,
                    &mut agent,
                    config,
                    RunMode::Train,
                    &device,
                )?;
            }
        },

        Env::Pointenv => {
            let mut env = *PointEnv::new(Default::default())?;
            let timelimit = *env.timelimit();
            let config = TrainingConfig::pointenv(timelimit);

            let size_state = env.observation_space().iter().product::<usize>();
            let size_action = env.action_space().iter().product::<usize>();

            let mut agent = DDPG::from_config(&device, &config, size_state, size_action)?;

            if args.gui {
                GUI::open(env, agent, config, device);
            } else {
                run(
                    &mut env,
                    &mut agent,
                    config,
                    RunMode::Train,
                    &device,
                )?;
            }
        },
    }
    Ok(())
}

