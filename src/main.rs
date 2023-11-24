use {
    anyhow::Result,
    graph_rl::{
        agents::{
            DDPG,
            Algorithm,
            OffPolicyAlgorithm,
            configs::{
                DDPGConfig,
                AlgorithmConfig,
                ActorCriticConfig,
                OffPolicyConfig,
                SgmConfig,
            },
        },
        envs::{
            Environment,
            DistanceMeasure,
            Sampleable,
            TensorConvertible,
            Renderable,

            PendulumEnv,

            PointEnv,
            PointEnvConfig,
            PointReward,

            PointMazeEnv,
        },
        engines::{
            run_experiment_off_policy,
            GUI,
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
    serde::Serialize,
    std::{
        fmt::Debug,
        hash::Hash,
    },
    tracing::Level,
    std::{
        fs::{File, create_dir_all},
        path::Path,
        sync::Arc,
    },
    tracing_subscriber::{
        fmt::{
            layer,
            writer::MakeWriterExt,
        },
        layer::SubscriberExt,
        util::SubscriberInitExt,
    },
};

pub fn setup_logging(
    path: &dyn AsRef<Path>,
    min_level_file: Option<Level>,
    min_level_stdout: Option<Level>,
) -> Result<()> {
    let path = Path::new("data/").join(path);
    create_dir_all(path.as_path())?;
    let log_file = Arc::new(File::create(path.join("debug.log"))?);

    tracing_subscriber::registry()
        // File writer
        .with(
            layer()
                .with_writer(log_file.with_max_level(match min_level_file {
                    Some(level) => level,
                    None => Level::INFO,
                }))
                .with_ansi(false),
        )
        // Stdout writer
        .with(
            layer()
                .with_writer(std::io::stdout.with_max_level(match min_level_stdout {
                    Some(level) => level,
                    None => Level::INFO,
                }))
                .compact()
                .pretty()
                .with_line_number(true)
                .with_thread_ids(false)
                .with_target(false),
        )
        // Create and set Subscriber
        .init();

    Ok(())
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

    /// The environment to run.
    #[arg(long, value_enum)]
    pub env: Env,

    /// Run as a GUI instead of just training.
    #[arg(long)]
    pub gui: bool,

    /// File to write the results to.
    #[arg(long)]
    pub output: Option<String>,

    /// Number of runs to collect data on.
    #[arg(long, default_value_t=10)]
    pub runs: usize,
}

/// This handles setup up logging, the GUI, and/or training, which simplifies
/// the main function, however it requires that the algorithm and environment
/// all implement every possible trait that we support.
///
/// When we add a new algorithm or environment which doesn't (yet) support e.g.
/// Renderable or ActorCriticConfig, we'll have to use the run_n() or GUI
/// methods directly like before.
pub fn do_stuff<Alg, Env, Obs, Act>(
    mut env: Env,
    config: Alg::Config,
    args: Args,
) -> Result<()>
where
    Alg: Algorithm + OffPolicyAlgorithm + 'static,
    Alg::Config: AlgorithmConfig + ActorCriticConfig + OffPolicyConfig + SgmConfig + Clone + Serialize,
    Env: Environment<Action = Act, Observation = Obs> + Renderable + 'static,
    Env::Config: Clone + Serialize + 'static,
    Obs: Debug + Clone + Eq + Hash + TensorConvertible + DistanceMeasure + 'static,
    Act: Clone + TensorConvertible + Sampleable + 'static,
{
    let name = if let Some(name) = args.output.clone() {
        name
    } else {
        args.env.name().to_owned()
    };

    setup_logging(
        &name,
        args.log.level(),
        args.log.level(),
    )?;

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::Cuda(CudaDevice::new(0)?)
    };

    if args.gui {
        let agent = *Alg::from_config(
            &device,
            &config,
            env.observation_space().iter().product::<usize>(),
            env.action_space().iter().product::<usize>(),
        )?;
        GUI::<Alg, Env, Obs, Act>::open(
            env,
            agent,
            config,
            device,
        );
    } else {
        run_experiment_off_policy::<Alg, Env, Obs, Act>(
            &name,
            args.runs,
            &mut env,
            config,
            &device,
        )?;
    }
    Ok(())
}

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
