pub mod logging;

pub mod envs;
pub mod components;
pub mod agents;

pub mod cli;
pub mod gui;
pub mod engine;


#[derive(Clone, Copy)]
pub enum RunMode {
    Train,
    Test,
}