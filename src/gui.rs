use std::{thread, time};

use crate::{
    ddpg::DDPG,
    envs::point_env::{
        PointEnv,
        PointState,
    },
    run,
    tick,
    TrainingConfig,
};
use candle_core::Device;
use ordered_float::OrderedFloat;
use petgraph::Graph;
use petgraph::visit::EdgeRef;

use eframe::egui;
use egui::widgets::plot::PlotUi;
use egui::widgets::Button;
use egui::plot::{Plot, Line, Points, PlotBounds};
use egui::{Ui, Slider, Color32};


pub struct GUI<'a> {
    env: PointEnv,
    agent: DDPG<'a>,
    config: TrainingConfig,
    device: Device,

    play: bool,
}
impl GUI<'static> {
    pub fn open(
        env: PointEnv,
        agent: DDPG<'static>,
        config: TrainingConfig,
        device: Device,
    ) {
        let gui = Self {
            env,
            agent,
            config,
            device,
            play: false,
        };
        eframe::run_native(
            "Actor-Critic Graph-Learner",
            eframe::NativeOptions::default(),
            Box::new(|_| Box::new(gui)),
        ).unwrap();
    }

    fn render_pointenv(
        &self,
        plot_ui: &mut PlotUi,
        plot_path: bool,
        plot_graph: bool,
    ) {
        // Setup plot bounds
        plot_ui.set_plot_bounds(
            PlotBounds::from_min_max(
                [0.0, 0.0],
                [*self.env.width() as f64, *self.env.height() as f64],
            )
        );
        // Plot walls
        for wall in self.env.walls().iter() {
            plot_ui.line(
                Line::new(
                    vec![
                        [wall.A.x(), wall.A.y()],
                        [wall.B.x(), wall.B.y()],
                    ]
                )
                .width(2.0)
                .color(Color32::WHITE)
            )
        }
        // Plot start and goal
        let start = self.env.start();
        plot_ui.points(
            Points::new(
                vec![
                    [start.x(), start.y()],
                ]
            )
            .radius(2.0)
            .color(Color32::WHITE)
        );
        let goal = self.env.goal();
        plot_ui.points(
            Points::new(
                vec![
                    [goal.x(), goal.y()],
                ]
            )
            .radius(2.0)
            .color(Color32::GREEN)
        );
        // Plot path
        if plot_path {
            plot_ui.line(
                Line::new(
                    self.env.history()
                    .iter()
                    .map(|p| {
                        [p.x(), p.y()]
                    })
                    .collect::<Vec<_>>()
                )
            )
        }
    }

    #[allow(dead_code)]
    fn plot_graph(
        graph: &Graph<PointState, OrderedFloat<f32>>,
        plot_ui: &mut PlotUi,
    ) {
        for edge in graph.edge_references() {
            let s1 = graph[edge.source()];
            let s2 = graph[edge.target()];

            plot_ui.line(
                Line::new(
                    vec![
                        [s1.x(), s1.y()],
                        [s2.x(), s2.y()],
                    ]
                )
                .width(1.0)
                .color(Color32::LIGHT_BLUE)
            )
        }
    }

    fn render_options(
        &mut self,
        ui: &mut Ui,
    ) {
        ui.heading("Settings");

        ui.separator();
        ui.label("DDPG Options");
        ui.add(Slider::new(&mut self.config.actor_learning_rate, 0.0001..=0.1).logarithmic(true).text("Actor LR"));
        ui.add(Slider::new(&mut self.config.critic_learning_rate, 0.0001..=0.1).logarithmic(true).text("Critic LR"));
        ui.add(Slider::new(&mut self.config.gamma, 0.0..=1.0).step_by(0.1).text("Gamma"));
        ui.add(Slider::new(&mut self.config.tau, 0.001..=1.0).logarithmic(true).text("Tau"));
        ui.add(Slider::new(&mut self.config.replay_buffer_capacity, 100..=100_000).logarithmic(true).text("Buffer size"));
        ui.add(Slider::new(&mut self.config.training_batch_size, 1..=200).text("Batch size"));
        ui.add(Slider::new(&mut self.config.training_iterations, 1..=200).text("Training Iterations"));

        ui.separator();
        ui.label("SGM Options");
        // ui.add(Slider::new(&mut alg.sgm_freq, 1..=11).text("sgm_freq").step_by(1.0));
        // ui.add(Slider::new(&mut alg.max_dist.0, 0.0..=1.0).text("max_dist").step_by(0.01));
        // ui.add(Slider::new(&mut alg.tau.0, 0.0..=1.0).text("tau").step_by(0.01));

        ui.separator();
        ui.label("Run Options");
        ui.add(Slider::new(&mut self.config.max_episodes, 1..=101).text("n_episodes"));
        if ui.add(Button::new("Run Episodes")).clicked() {
            run(
                &mut self.env,
                &mut self.agent,
                self.config.clone(),
                true,
                &self.device,
            ).unwrap();
        };
        ui.horizontal(|ui| {
            if ui.add(Button::new("Play")).clicked() {
                self.play = true;
            };
            if ui.add(Button::new("Pause")).clicked() {
                self.play = false;
            };
        });
        if self.play {
            tick(
                &mut self.env,
                &mut self.agent,
                self.config.clone(),
                false,
                &self.device,
            ).unwrap();
        }
    }
}

impl eframe::App for GUI<'static> {
    fn update(
        &mut self,
        ctx: &egui::Context,
        _frame: &mut eframe::Frame,
    ) {
        // render the gui
        egui::SidePanel::right("side_panel").show(ctx, |ui| {
            self.render_options(ui);
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            Plot::new("main_panel").show(ui, |plot_ui| { //.view_aspect(1.0)
                self.render_pointenv(plot_ui, true, true);
            });
        });

        // sleep for a bit
        thread::sleep(time::Duration::from_millis(100));

        // always repaint, not just on mouse-hover
        ctx.request_repaint();
    }
}
