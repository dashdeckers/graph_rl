use {
    anyhow::Result,
    candle_core::Device,
    graph_rl::{
        agents::{
            DDPG,
            DDPGConfig,
        },
        envs::{
            Environment,
            PendulumEnv,

            PointEnv,
            PointEnvConfig,
            PointReward,

            PointMazeEnv,
        },
        cli::{
            Args,
            Env,
            do_stuff,
        },
    },
    clap::Parser,
};


// NOW

// >- Distributional RL

// LATER

// >- Add Cuda as a Device and get that working on the server
// >- Put the Candle "cuda" feature behind a cfg() flag
//    `-> https://doc.rust-lang.org/cargo/reference/features.html


fn main() -> Result<()> {
    let args = Args::parse();
    let experiment_name = if let Some(name) = args.output.clone() {
        name
    } else {
        args.env.name().to_owned()
    };

    // let device = &if args.cpu { Device::Cpu } else { Device::Cpu }; // TODO: Cuda
    let device = Device::Cpu;
    match args.env {
        Env::Pendulum => {
            do_stuff::<DDPG, PendulumEnv, _, _>(
                *PendulumEnv::new(Default::default())?,
                DDPGConfig::pendulum(),
                device,
                args.clone(),
                &experiment_name,
            )?;
        }

        Env::Pointenv => {
            do_stuff::<DDPG, PointEnv, _, _>(
                *PointEnv::new(PointEnvConfig::new(
                    5,
                    5,
                    None,
                    30,
                    1.0,
                    0.5,
                    0.1,
                    PointReward::Distance,
                    42,
                ))?,
                DDPGConfig::pointenv(),
                device,
                args.clone(),
                &experiment_name,
            )?;
        }

        Env::Pointmaze => {
            do_stuff::<DDPG, PointMazeEnv, _, _>(
                *PointMazeEnv::new(Default::default())?,
                DDPGConfig::pointmaze(),
                device,
                args.clone(),
                &experiment_name,
            )?;
        }
    }
    Ok(())
}
