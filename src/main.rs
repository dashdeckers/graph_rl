extern crate intel_mkl_src;

use candle_core::{Device, Tensor};
use clap::Parser;
use rand::Rng;
use anyhow::Result;
use tracing::Level;

#[allow(unused_imports)]
use graph_rl::{
    ddpg::DDPG,
    ou_noise::OuNoise,
    envs::{
        pendulum::PendulumEnv,
        point_env::PointEnv,
        Environment,
    },
    util::setup_logging,
    gui::GUI,
};

// The impact of the q value of the next state on the current state's q value.
const GAMMA: f64 = 0.99;
// The weight for updating the target networks.
const TAU: f64 = 0.005;
// The capacity of the replay buffer used for sampling training data.
const REPLAY_BUFFER_CAPACITY: usize = 100_000;
// The training batch size for each training iteration.
const TRAINING_BATCH_SIZE: usize = 100;
// The total number of episodes.
const MAX_EPISODES: usize = 20;// 100;
// The maximum length of an episode.
const EPISODE_LENGTH: usize = 200;
// The number of training iterations after one episode finishes.
const TRAINING_ITERATIONS: usize = 200;

// Ornstein-Uhlenbeck process parameters.
const MU: f64 = 0.0;
const THETA: f64 = 0.15;
const SIGMA: f64 = 0.1;

const ACTOR_LEARNING_RATE: f64 = 1e-4;
const CRITIC_LEARNING_RATE: f64 = 1e-3;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,
}


// ChromeTracing?
// A2C?
// CUDA?

// 1. Make PointEnv API compatible with (= the same as) Gymnasium (also cleanup)
// 2. Run DDPG on PointEnv (observe learning?)
// 3. Get the GUI back up and running
// 4. Get SGM working, visualize it (each env-type gets its own State struct!)





fn main() -> Result<()> {
    // use tracing_chrome::ChromeLayerBuilder;
    // use tracing_subscriber::prelude::*;

    // let args = Args::parse();

    // let _guard = if args.tracing {
    //     let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
    //     tracing_subscriber::registry().with(chrome_layer).init();
    //     Some(guard)
    // } else {
    //     None
    // };


    setup_logging(
        "debug.log".into(),
        Some(Level::INFO),
        Some(Level::WARN),
    )?;

    let mut env = *PointEnv::new(Default::default())?;
    // let env = *PendulumEnv::new(Default::default())?;

    let size_state = env.observation_space().iter().product::<usize>();
    let size_action = env.action_space();

    let mut agent = DDPG::new(
        &Device::Cpu,
        size_state,
        size_action,
        true,
        ACTOR_LEARNING_RATE,
        CRITIC_LEARNING_RATE,
        GAMMA,
        TAU,
        REPLAY_BUFFER_CAPACITY,
        OuNoise::new(MU, THETA, SIGMA, size_action)?,
    )?;

    // let postprocess_action = |actions: &[f64]| {
    //     actions
    //         .iter()
    //         .map(|a| (a * 2.0).clamp(-2.0, 2.0))
    //         .collect::<Vec<f64>>()
    // };
    let postprocess_action = |actions: &[f64]| {
        actions.to_vec()
    };

    run(
        &mut env,
        &mut agent,
        MAX_EPISODES,
        EPISODE_LENGTH,
        true,
        postprocess_action,
    )?;

    GUI::show(GUI::new(env, agent));

    Ok(())
}


fn run<E: Environment>(
    env: &mut E,
    agent: &mut DDPG,
    max_episodes: usize,
    episode_length: usize,
    train: bool,
    postprocess_action: fn(&[f64]) -> Vec<f64>,
) -> Result<()> {

    println!("action space: {}", env.action_space());
    println!("observation space: {:?}", env.observation_space());


    let mut rng = rand::thread_rng();

    agent.train = train;
    for episode in 0..max_episodes {
        // let mut state = env.reset(episode as u64)?;
        let mut state = env.reset(rng.gen::<u64>())?;
        let mut total_reward = 0.0;

        for _ in 0..episode_length {
            let action = agent.actions(&state.clone().into())?;
            let action = postprocess_action(&action);

            let step = env.step(action.clone().into())?;
            total_reward += step.reward;

            if train {
                agent.remember(
                    &state.into(),
                    &Tensor::new(action, &Device::Cpu)?,
                    &Tensor::new(vec![step.reward], &Device::Cpu)?,
                    &step.state.clone().into(),
                    step.terminated,
                    step.truncated,
                );
            }

            if step.terminated || step.truncated {
                break;
            }
            state = step.state;
        }

        println!("episode {episode} with total reward of {total_reward}");

        if train {
            for _ in 0..TRAINING_ITERATIONS {
                agent.train(TRAINING_BATCH_SIZE)?;
            }
        }
    }
    Ok(())
}