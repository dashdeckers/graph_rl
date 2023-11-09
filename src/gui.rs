use std::{thread, time};
use std::hash::Hash;
use std::fmt::Debug;

use crate::{
    ddpg::DDPG,
    envs::{
        Renderable,
        Environment,
        VectorConvertible,
        TensorConvertible,
        DistanceMeasure,
        Sampleable,
    },
    TrainingConfig,
    RunMode,
    run,
    tick,
};
use anyhow::Result;
use candle_core::Device;
use ordered_float::OrderedFloat;
use petgraph::{Undirected, stable_graph::StableGraph};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};

use eframe::egui;
use egui::widgets::Button;
use egui::{Ui, Slider, Color32, Checkbox};
use egui_graphs::{Graph, GraphView, SettingsInteraction, SettingsStyle};
use egui_plot::{PlotUi, Plot, Line, Points, PlotBounds};


enum PlayMode {
    Pause,
    Ticks,
    Episodes,
}

pub struct GUI<'a, E, O, A>
where
    E: Environment<Action = A, Observation = O> + Renderable + 'static,
    O: Debug + Clone + Eq + Hash + TensorConvertible + DistanceMeasure + 'static,
    A: Clone + VectorConvertible + Sampleable + 'static,
{
    env: E,
    agent: DDPG<'a>,
    config: TrainingConfig,
    device: Device,

    run_data: Vec<(RunMode, f64, bool)>,
    graph: StableGraph<O, OrderedFloat<f64>, Undirected>,
    graph_egui: Graph<O, OrderedFloat<f64>, Undirected>,

    play_mode: PlayMode,
    last_graph_constructed_at: usize,

    render_graph: bool,
    render_buffer: bool,
    render_fancy_graph: bool,
}

impl<E, O, A> eframe::App for GUI<'static, E, O, A>
where
    E: Environment<Action = A, Observation = O> + Renderable + 'static,
    O: Debug + Clone + Eq + Hash + TensorConvertible + DistanceMeasure + 'static,
    A: Clone + VectorConvertible + Sampleable + 'static,
{
    fn update(
        &mut self,
        ctx: &egui::Context,
        _frame: &mut eframe::Frame,
    ) {
        // run the GUI logic
        self.run_gui_logic().unwrap();

        // render the settings and options
        egui::SidePanel::left("settings").show(ctx, |ui| {
            self.render_options(ui);
        });

        // render episodic rewards / learning curve
        egui::TopBottomPanel::top("rewards").show(ctx, |ui| {
            Plot::new("rewards_plot").show(ui, |plot_ui| {
                self.render_returns(plot_ui);
            });
        });

        // render the bottom panel in which we show the clicked on node's state
        egui::TopBottomPanel::bottom("inspect").show(ctx, |ui| {
            let node = self.graph_egui.g.node_indices().find(|e| self.graph_egui.g[*e].selected());
            if let Some(node) = node {
                ui.label(format!("Selected node: {:#?}", self.graph_egui.g[node].data()));
            } else {
                ui.label("No node selected.");
            }
        });

        // render the environment / graph
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.render_fancy_graph {
                self.render_fancy_graph(ui);
            } else {
                Plot::new("environment").show(ui, |plot_ui| { //.view_aspect(1.0)
                    self.env.render(plot_ui);
                    if self.render_graph {
                        self.render_graph(plot_ui);
                    }
                    if self.render_buffer {
                        self.render_buffer(plot_ui);
                    }
                });
            }
        });

        // sleep for a bit
        thread::sleep(time::Duration::from_millis(100));

        // always repaint, not just on mouse-hover
        ctx.request_repaint();
    }
}

impl<E, O, A> GUI<'static, E, O, A>
where
    E: Environment<Action = A, Observation = O> + Renderable + 'static,
    O: Debug + Clone + Eq + Hash + TensorConvertible + DistanceMeasure + 'static,
    A: Clone + VectorConvertible + Sampleable + 'static,
{
    pub fn open(
        env: E,
        agent: DDPG<'static>,
        config: TrainingConfig,
        device: Device,
    ) {
        let gui = Self {
            env,
            agent,
            config,
            device,

            run_data: Vec::new(),
            graph: StableGraph::default(),
            graph_egui: Graph::from(&StableGraph::default()),

            play_mode: PlayMode::Pause,
            last_graph_constructed_at: 0,

            render_graph: false,
            render_buffer: false,
            render_fancy_graph: false,
        };
        eframe::run_native(
            "Actor-Critic Graph-Learner",
            eframe::NativeOptions{
                min_window_size: Some(egui::vec2(800.0, 600.0)),
                ..Default::default()
            },
            Box::new(|_| Box::new(gui)),
        ).unwrap();
    }

    fn run_gui_logic(
        &mut self,
    ) -> Result<()> {
        #[allow(clippy::collapsible_if)]
        // construct the graph every defined number of episodes
        if self.config.sgm_freq > 0 && (self.run_data.len() + 1) % self.config.sgm_freq == 0 {
            // but take care not to construct it constantly in this edge-case
            if self.last_graph_constructed_at != self.run_data.len() {
                self.graph = self.agent.replay_buffer().construct_sgm(
                    |o1, o2| <O>::distance(o1, o2),
                    self.config.sgm_maxdist,
                    self.config.sgm_tau,
                ).0;
                self.graph_egui = Graph::from(&self.graph);
                self.last_graph_constructed_at = self.run_data.len();
            }
        }

        // let it play to observe agent behavior!
        match self.play_mode {
            PlayMode::Pause => (),
            PlayMode::Ticks => {
                tick(
                    &mut self.env,
                    &mut self.agent,
                    &self.device,
                )?;
            }
            PlayMode::Episodes => {
                let run_mode = RunMode::Test;
                let mut config = self.config.clone();
                config.max_episodes = 1;
                let (mc_returns, successes) = run(
                    &mut self.env,
                    &mut self.agent,
                    config,
                    run_mode,
                    &self.device,
                )?;
                self.run_data.push((run_mode, mc_returns[0], successes[0]));
            }
        }
        Ok(())
    }

    fn reset_agent(
        &mut self,
    ) -> Result<()> {
        let size_state = self.env.observation_space().iter().product::<usize>();
        let size_action = self.env.action_space().iter().product::<usize>();
        self.agent = DDPG::from_config(&self.device, &self.config, size_state, size_action)?;
        Ok(())
    }

    fn render_graph(
        &self,
        plot_ui: &mut PlotUi,
    ) {
        for edge in self.graph.edge_references() {
            let s1 = &self.graph[edge.source()];
            let s2 = &self.graph[edge.target()];

            let s1 = <O>::to_vec(s1.clone());
            let s2 = <O>::to_vec(s2.clone());

            plot_ui.line(
                Line::new(
                    vec![
                        [s1[0], s1[1]],
                        [s2[0], s2[1]],
                    ]
                )
                .width(1.0)
                .color(Color32::LIGHT_BLUE)
            )
        }
    }

    fn render_buffer(
        &self,
        plot_ui: &mut PlotUi,
    ) {
        for state in self.agent.replay_buffer().all_states::<O>() {
            let s = <O>::to_vec(state.clone());
            plot_ui.points(
                Points::new(
                    vec![
                        [s[0], s[1]],
                    ]
                )
                .radius(2.0)
                .color(Color32::RED)
            );
        }
    }

    fn render_returns(
        &mut self,
        plot_ui: &mut PlotUi,
    ) {
        let (min, max) = self.env.value_range();
        let n = self.run_data.len() as f64;

        plot_ui.set_plot_bounds(
            PlotBounds::from_min_max(
                [0.0, min],
                [n, max],
            )
        );

        let _ = self.run_data
            .iter()
            .enumerate()
            .map(|(idx, (run_mode, mc_return, success))|{
                plot_ui.points(
                    Points::new([idx as f64, *mc_return])
                        .color(match run_mode {
                            RunMode::Train => Color32::RED,
                            RunMode::Test => Color32::GREEN,
                        })
                );
                if *success {
                    plot_ui.line(
                        Line::new(
                            vec![
                                [idx as f64, min],
                                [idx as f64, max],
                            ]
                        )
                        .width(1.0)
                        .color(Color32::LIGHT_GREEN)
                    )
                }
            })
            .collect::<Vec<_>>();
    }

    fn render_fancy_graph(
        &mut self,
        ui: &mut Ui,
    ) {
        let interaction_settings = &SettingsInteraction::new()
            .with_dragging_enabled(true)
            .with_clicking_enabled(true)
            .with_selection_enabled(true);
        let style_settings = &SettingsStyle::new().with_labels_always(true);
        ui.add(
            &mut GraphView::new(&mut self.graph_egui)
                .with_styles(style_settings)
                .with_interactions(interaction_settings),
        );
    }

    fn render_options(
        &mut self,
        ui: &mut Ui,
    ) {
        ui.heading("Settings");

        ui.separator();
        ui.label("DDPG Options");
        ui.add(Slider::new(&mut self.config.actor_learning_rate, 0.00001..=0.1).logarithmic(true).fixed_decimals(5).text("Actor LR"));
        ui.add(Slider::new(&mut self.config.critic_learning_rate, 0.00001..=0.1).logarithmic(true).fixed_decimals(5).text("Critic LR"));
        ui.add(Slider::new(&mut self.config.gamma, 0.0..=1.0).step_by(0.1).text("Gamma"));
        ui.add(Slider::new(&mut self.config.tau, 0.001..=1.0).logarithmic(true).text("Tau"));
        ui.add(Slider::new(&mut self.config.replay_buffer_capacity, 10..=100_000).logarithmic(true).text("Buffer size"));
        ui.add(Slider::new(&mut self.config.training_batch_size, 1..=200).text("Batch size"));
        ui.add(Slider::new(&mut self.config.training_iterations, 1..=200).text("Training iters"));

        ui.separator();
        ui.label("SGM Options");
        ui.add(Slider::new(&mut self.config.sgm_freq, 0..=20).text("Rebuilding freq").step_by(1.0));
        ui.add(Slider::new(&mut self.config.sgm_maxdist, 0.0..=1.0).text("Max distance").step_by(0.01));
        ui.add(Slider::new(&mut self.config.sgm_tau, 0.0..=1.0).text("Tau").step_by(0.01));

        ui.separator();
        ui.label("Train Agent");
        ui.add(Slider::new(&mut self.config.max_episodes, 1..=101).text("n_episodes"));
        ui.horizontal(|ui| {
            if ui.add(Button::new("Train Episodes")).clicked() {
                let (mc_returns, successes) = run(
                    &mut self.env,
                    &mut self.agent,
                    self.config.clone(),
                    RunMode::Train,
                    &self.device,
                ).unwrap();
                self.run_data.extend(
                    (0..self.config.max_episodes)
                    .map(|i| (RunMode::Train, mc_returns[i], successes[i]))
                );
            };
            if ui.add(Button::new("Reset Agent")).clicked() {
                self.reset_agent().unwrap();
            };
        });

        ui.separator();
        ui.label("Test Agent");
        ui.horizontal(|ui| {
            if ui.add(Button::new("Pause")).clicked() {
                self.play_mode = PlayMode::Pause;
            };
            if ui.add(Button::new("Play(t)")).clicked() {
                self.play_mode = PlayMode::Ticks;
            };
            if ui.add(Button::new("Play(e)")).clicked() {
                self.play_mode = PlayMode::Episodes;
            };
        });

        ui.separator();
        ui.label("Render Options");
        ui.add(Checkbox::new(&mut self.render_graph, "Show Graph"));
        ui.add(Checkbox::new(&mut self.render_buffer, "Show Buffer"));
        ui.add(Checkbox::new(&mut self.render_fancy_graph, "Fancy GraphView"));
    }
}
