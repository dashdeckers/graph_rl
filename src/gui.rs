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
    },
    TrainingConfig,
    run,
    tick,
};
use candle_core::Device;
use ordered_float::OrderedFloat;
use petgraph::{Directed, stable_graph::StableGraph};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};

use eframe::egui;
use egui::widgets::Button;
use egui::{Ui, Slider, Color32, Checkbox};
use egui_graphs::{Graph, GraphView};
use egui_plot::{PlotUi, Plot, Line, Points};


enum PlayMode {
    Pause,
    Ticks,
    Episodes,
}

pub struct GUI<'a, E, O, A>
where
    E: Environment<Action = A, Observation = O> + Renderable + 'static,
    O: Debug + Clone + Eq + Hash + TensorConvertible + DistanceMeasure + 'static,
    A: Clone + VectorConvertible + 'static,
{
    env: E,
    agent: DDPG<'a>,
    config: TrainingConfig,
    device: Device,

    episodic_returns: Vec<f64>,
    graph: StableGraph<O, OrderedFloat<f64>, Directed>,

    play_mode: PlayMode,
    train_episodes_to_go: usize,
    last_graph_constructed_at: usize,

    render_graph: bool,
    render_buffer: bool,
    render_fancy_graph: bool,
}
impl<E, O, A> GUI<'static, E, O, A>
where
    E: Environment<Action = A, Observation = O> + Renderable + 'static,
    O: Debug + Clone + Eq + Hash + TensorConvertible + DistanceMeasure + 'static,
    A: Clone + VectorConvertible + 'static,
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

            episodic_returns: Vec::new(),
            graph: StableGraph::new(),

            play_mode: PlayMode::Pause,
            train_episodes_to_go: 0,
            last_graph_constructed_at: 0,

            render_graph: false,
            render_buffer: false,
            render_fancy_graph: false,
        };
        eframe::run_native(
            "Actor-Critic Graph-Learner",
            eframe::NativeOptions::default(),
            Box::new(|_| Box::new(gui)),
        ).unwrap();
    }

    fn run_gui_logic(
        &mut self,
    ) {
        // a kind of hack not to hang up the GUI while training and watch it train one episode at a time
        if self.train_episodes_to_go > 0 {
            let mut config = self.config.clone();
            config.max_episodes = 1;
            self.episodic_returns.extend(run(
                &mut self.env,
                &mut self.agent,
                config,
                true,
                &self.device,
            ).unwrap());
            self.train_episodes_to_go -= 1;
        }

        #[allow(clippy::collapsible_if)]
        // construct the graph every defined number of episodes
        if (self.episodic_returns.len() + 1) % self.config.sgm_freq == 0 {
            // but take care not to construct it constantly in this edge-case
            if self.last_graph_constructed_at != self.episodic_returns.len() {
                self.graph = self.agent.replay_buffer().construct_sgm(
                    |o1, o2| <O>::distance(o1, o2),
                    self.config.sgm_maxdist,
                    self.config.sgm_tau,
                ).0;
                self.last_graph_constructed_at = self.episodic_returns.len();
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
                ).unwrap();
            }
            PlayMode::Episodes => {
                let mut config = self.config.clone();
                config.max_episodes = 1;
                run(
                    &mut self.env,
                    &mut self.agent,
                    config,
                    false,
                    &self.device,
                ).unwrap();
            }
        }
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

    fn render_rewards(
        &mut self,
        ui: &mut Ui,
    ) {
        Plot::new("data_plot").show(ui, |plot_ui| {
            plot_ui.line(
                Line::new(
                    self.episodic_returns.clone()
                    .into_iter()
                    .enumerate()
                    .map(|(x, y)| [x as f64, y])
                    .collect::<Vec<_>>()
                )
            )
        });
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
        ui.add(Slider::new(&mut self.config.training_iterations, 1..=200).text("Training iters"));

        ui.separator();
        ui.label("SGM Options");
        ui.add(Slider::new(&mut self.config.sgm_freq, 1..=20).text("Rebuilding freq").step_by(1.0));
        ui.add(Slider::new(&mut self.config.sgm_maxdist, 0.0..=1.0).text("Max distance").step_by(0.01));
        ui.add(Slider::new(&mut self.config.sgm_tau, 0.0..=1.0).text("Tau").step_by(0.01));

        ui.separator();
        ui.label("Train Agent");
        ui.add(Slider::new(&mut self.config.max_episodes, 1..=101).text("n_episodes"));
        if ui.add(Button::new("Train Episodes")).clicked() {
            self.train_episodes_to_go = self.config.max_episodes;
        };

        ui.separator();
        ui.label("Watch Agent");
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

impl<E, O, A> eframe::App for GUI<'static, E, O, A>
where
    E: Environment<Action = A, Observation = O> + Renderable + 'static,
    O: Debug + Clone + Eq + Hash + TensorConvertible + DistanceMeasure + 'static,
    A: Clone + VectorConvertible + 'static,
{
    fn update(
        &mut self,
        ctx: &egui::Context,
        _frame: &mut eframe::Frame,
    ) {
        // run the GUI logic
        self.run_gui_logic();

        // render the settings and options
        egui::SidePanel::left("settings").show(ctx, |ui| {
            self.render_options(ui);
        });

        // render episodic rewards / learning curve
        egui::TopBottomPanel::top("rewards").show(ctx, |ui| {
            self.render_rewards(ui);
        });

        // render the environment / graph
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.render_fancy_graph {
                ui.add(&mut GraphView::new(&mut Graph::from(&self.graph)));
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
