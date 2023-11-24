//! Some doc text
#![ doc=mermaid!( "../docs/example.mmd" ) ]

use simple_mermaid::mermaid;

pub mod envs;
pub mod components;
pub mod agents;
pub mod engines;

/// The execution mode of an agent is either training or testing.
#[derive(Clone, Copy)]
pub enum RunMode {
    Train,
    Test,
}