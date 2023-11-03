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
    gui::GUI,
    TrainingConfig,
    run,
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




// >- Render / Visualize SGM graphs
//    `-> maybe get VNC going on server
//    `-> egui graphs: https://github.com/blitzarx1/egui_graphs
//    `-> clickable nodes --> show / render that observation!
//    `-> plot the episodic returns as well in a bottom-plot




// >- Add AntMaze / PointMaze as Goal-Aware Environments
//    `-> these return a dict with 3 keys of Vec<64> instead of just a Vec<f64>
//    `-> also implement PlottableEnvironment for Pendulum!

// >- Add Cuda as a Device and get that working on the server
// >- Put the Candle "cuda" feature behind a cfg() flag
//    `-> https://doc.rust-lang.org/cargo/reference/features.html


fn main() -> Result<()> {
    let args = Args::parse();
    if args.logging {
        setup_logging(
            "debug.log".into(),
            Some(Level::WARN),
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
            let mut config = TrainingConfig::pointenv(timelimit);

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
                GUI::open(env, agent, config, device);
            } else {
                run(
                    &mut env,
                    &mut agent,
                    config.clone(),
                    true,
                    &device,
                )?;

                println!("Testing:");
                config.max_episodes = 10;
                run(
                    &mut env,
                    &mut agent,
                    config,
                    false,
                    &device,
                )?;
            }
        },
    }
    Ok(())
}

