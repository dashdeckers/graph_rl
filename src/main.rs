use {
    anyhow::Result,
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

    match args.env {
        Env::Pendulum => {
            do_stuff::<DDPG, PendulumEnv, _, _>(
                *PendulumEnv::new(Default::default())?,
                DDPGConfig::pendulum(),
                args.clone(),
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
                args.clone(),
            )?;
        }

        Env::Pointmaze => {
            do_stuff::<DDPG, PointMazeEnv, _, _>(
                *PointMazeEnv::new(Default::default())?,
                DDPGConfig::pointmaze(),
                args.clone(),
            )?;
        }
    }
    Ok(())
}
