use {
    crate::{
        agents::{
            Algorithm,
            AlgorithmConfig,
            ActorCriticConfig,
            OffPolicyAlgorithm,
            OffPolicyConfig,
            SgmConfig,
        },
        envs::{
            DistanceMeasure,
            Environment,
            Sampleable,
            TensorConvertible,
            Renderable,
        },
        engine::run_n,
        gui::GUI,
        logging::setup_logging,
    },
    anyhow::Result,
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
};


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
        run_n::<Alg, Env, Obs, Act>(
            &name,
            args.runs,
            &mut env,
            config,
            &device,
        )?;
    }
    Ok(())
}