#![allow(rustdoc::private_intra_doc_links)]
//! Some doc text
#![ doc=mermaid!( "../docs/example.mmd" ) ]

use simple_mermaid::mermaid;

pub mod envs;
pub mod components;
pub mod agents;
pub mod engines;
pub mod configs;


pub mod util {
    use anyhow::Result;
    use serde::{Serialize, Deserialize};
    use std::{
        fs::File,
        io::{Read, Write},
        path::Path,
    };

    pub fn write_config<C: Serialize, P: AsRef<Path>>(config: &C, path: P) -> Result<()> {
        Ok(File::create(path)?.write_all(
            ron::ser::to_string_pretty(
                config,
                ron::ser::PrettyConfig::default(),
            )?.as_bytes()
        )?)
    }

    pub fn read_config<C: for<'a> Deserialize<'a>, P: AsRef<Path>>(path: P) -> Result<C> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let result: C = ron::from_str(&contents)?;
        Ok(result)
    }
}


pub mod cli {
    use clap::{
        Parser,
        ValueEnum,
    };

    #[derive(ValueEnum, Debug, Clone)]
    pub enum ArgLoglevel {
        Error,
        Warn,
        Info,
        None,
    }

    #[derive(ValueEnum, Debug, Clone)]
    pub enum ArgDevice {
        Cpu,
        Cuda,
    }

    #[derive(Parser, Debug, Clone)]
    #[command(author, version, about, long_about = None)]
    pub struct Args {
        /// Device to run on (e.g. CPU / GPU).
        #[arg(long, value_enum, default_value_t=ArgDevice::Cpu)]
        pub device: ArgDevice,

        /// Pretrain the model according to the given config.
        #[arg(long)]
        pub pretrain_train_config: Option<String>,

        /// Pretrain the model on an Environment with the given config.
        #[arg(long)]
        pub pretrain_env_config: Option<String>,

        /// Train the model according to the given config.
        #[arg(long)]
        pub train_config: Option<String>,

        /// Environment config.
        #[arg(long)]
        pub env_config: Option<String>,

        /// Algorithm config.
        #[arg(long)]
        pub alg_config: Option<String>,

        /// Load a pretrained model from a file.
        #[arg(long, num_args = 2)]
        pub load_model: Option<Vec<String>>,

        /// Setup logging
        #[arg(long, value_enum, default_value_t=ArgLoglevel::Warn)]
        pub log: ArgLoglevel,

        /// Experiment name to use for logging / collecting data.
        #[arg(long)]
        pub name: String,

        /// Run as a GUI instead of just training.
        #[arg(long, default_value_t=false)]
        pub gui: bool,

        /// Number of repetitions for the experiment.
        #[arg(long, default_value_t=10)]
        pub n_repetitions: usize,
    }
}