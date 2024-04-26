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