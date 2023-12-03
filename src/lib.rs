#![allow(rustdoc::private_intra_doc_links)]
//! Some doc text
#![ doc=mermaid!( "../docs/example.mmd" ) ]

use simple_mermaid::mermaid;

pub mod envs;
pub mod components;
pub mod agents;
pub mod engines;


use std::fmt::Display;

/// The execution mode of an agent is either training or testing.
#[derive(Clone, Copy)]
pub enum RunMode {
    Train,
    Test,
}

impl Display for RunMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunMode::Train => write!(f, "Train"),
            RunMode::Test => write!(f, "Test"),
        }
    }
}