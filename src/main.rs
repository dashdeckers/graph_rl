extern crate intel_mkl_src;

use candle_core::Device;
use anyhow::Result;
use tracing::Level;
use clap::{Parser, ValueEnum};

use graph_rl::{
    ddpg::DDPG,
    ou_noise::OuNoise,
    envs::{
        PendulumEnv,
        PointEnv,
        Environment,
    },
    util::setup_logging,
    TrainingConfig,
    run,
    // gui::GUI,
};


#[derive(ValueEnum, Debug, Clone)]
enum Env {
    Pendulum,
    Pointenv,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Setup logging
    #[arg(long)]
    logging: bool,

    /// The environment to run.
    #[arg(value_enum)]
    env: Env,

    /// Run as a GUI instead of just training.
    #[arg(long)]
    gui: bool,
}


// >- Make SGM work on PointEnv!
//    `-> maybe get VNC going on server

// >- Add AntMaze / PointMaze as Goal-Aware Environments
//    `-> these return a dict with 3 keys of Vec<64> instead of just a Vec<f64>

// >- Add Cuda as a Device and get that working on the server
// >- Put the Candle "cuda" feature behind a cfg() flag
//    `-> https://doc.rust-lang.org/cargo/reference/features.html

// >- Find a way to render single observations so we can debug / viz SGM graphs
//    `-> egui graphs: https://github.com/blitzarx1/egui_graphs
//    `-> clickable nodes --> show the observation!


fn main() -> Result<()> {
    let args = Args::parse();
    if args.logging {
        setup_logging(
            "debug.log".into(),
            Some(Level::INFO),
            Some(Level::WARN),
        )?;
    }
    // let device = &if args.cpu { Device::Cpu } else { Device::Cpu }; // TODO: Cuda
    let device = Device::Cpu;
    match args.env {
        Env::Pendulum => {
            let mut env = *PendulumEnv::new(Default::default())?;
            let config = TrainingConfig::pendulum();

            let size_state = env.observation_space().iter().product::<usize>();
            let size_action = env.action_space().iter().product::<usize>();

            let mut agent = DDPG::new(
                &device,
                size_state,
                size_action,
                true,
                config.actor_learning_rate,
                config.critic_learning_rate,
                config.gamma,
                config.tau,
                config.replay_buffer_capacity,
                OuNoise::new(config.ou_mu, config.ou_theta, config.ou_sigma, size_action)?,
            )?;

            if args.gui {
                panic!("Not implemented yet!")
            } else {
                run(
                    &mut env,
                    &mut agent,
                    config,
                    true,
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

            let mut agent = DDPG::new(
                &device,
                size_state,
                size_action,
                true,
                config.actor_learning_rate,
                config.critic_learning_rate,
                config.gamma,
                config.tau,
                config.replay_buffer_capacity,
                OuNoise::new(config.ou_mu, config.ou_theta, config.ou_sigma, size_action)?,
            )?;

            if args.gui {
                panic!("Not implemented yet!")
                // GUI::open(env, agent, config, device);
            } else {
                run(
                    &mut env,
                    &mut agent,
                    config,
                    true,
                    &device,
                )?;
            }
        },
    }
    Ok(())
}

