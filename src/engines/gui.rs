use {
    super::{
        tick,
        training_loop_off_policy,
    },
    crate::{
        agents::{
            Algorithm,
            OffPolicyAlgorithm,
            configs::{
                AlgorithmConfig,
                ActorCriticConfig,
                OffPolicyConfig,
                SgmConfig,
            },
        },
        envs::{
            DistanceMeasure,
            Environment,
            Renderable,
            Sampleable,
            TensorConvertible,
        },
        RunMode,
    },
    anyhow::Result,
    candle_core::Device,
    eframe::egui,
    egui::{
        widgets::Button,
        Checkbox,
        Color32,
        Slider,
        Ui,
    },
    egui_graphs::{
        Graph,
        GraphView,
        SettingsInteraction,
        SettingsStyle,
    },
    egui_plot::{
        Line,
        Plot,
        PlotBounds,
        PlotUi,
        Points,
    },
    ordered_float::OrderedFloat,
    petgraph::{
        stable_graph::StableGraph,
        visit::{
            EdgeRef,
            IntoEdgeReferences,
        },
        Undirected,
    },
    std::{
        fmt::Debug,
        hash::Hash,
        thread,
        time,
    },
};

enum PlayMode {
    Pause,
    Ticks,
    Episodes,
}

pub struct GUI<Alg, Env, Obs, Act>
where
    Alg: Algorithm + OffPolicyAlgorithm,
    Alg::Config: AlgorithmConfig + ActorCriticConfig + OffPolicyConfig + SgmConfig + Clone,
    Env: Environment<Action = Act, Observation = Obs> + Renderable + 'static,
    Obs: Debug + Clone + Eq + Hash + TensorConvertible + DistanceMeasure + 'static,
    Act: Clone + TensorConvertible + Sampleable + 'static,
{
    env: Env,
    agent: Alg,
    config: Alg::Config,
    device: Device,

    run_data: Vec<(RunMode, f64, bool)>,
    graph: StableGraph<Obs, OrderedFloat<f64>, Undirected>,
    graph_egui: Graph<Obs, OrderedFloat<f64>, Undirected>,

    play_mode: PlayMode,
    last_graph_constructed_at: usize,

    render_graph: bool,
    render_buffer: bool,
    render_fancy: bool,
}

impl<Alg, Env, Obs, Act> eframe::App for GUI<Alg, Env, Obs, Act>
where
    Alg: Algorithm + OffPolicyAlgorithm + 'static,
    Alg::Config: AlgorithmConfig + ActorCriticConfig + OffPolicyConfig + SgmConfig + Clone,
    Env: Environment<Action = Act, Observation = Obs> + Renderable + 'static,
    Obs: Debug + Clone + Eq + Hash + TensorConvertible + DistanceMeasure + 'static,
    Act: Clone + TensorConvertible + Sampleable + 'static,
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
            let node = self
                .graph_egui
                .g
                .node_indices()
                .find(|e| self.graph_egui.g[*e].selected());

            if let Some(node) = node {
                ui.label(format!(
                    "Selected node: {:#?}",
                    self.graph_egui.g[node].data()
                ));
            } else {
                ui.label("No node selected.");
            }
        });

        // render the environment / graph
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.render_fancy {
                self.render_fancy(ui);
            } else {
                Plot::new("environment").show(ui, |plot_ui| {
                    //.view_aspect(1.0)
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

impl<Alg, Env, Obs, Act> GUI<Alg, Env, Obs, Act>
where
    Alg: Algorithm + OffPolicyAlgorithm + 'static,
    Alg::Config: AlgorithmConfig + ActorCriticConfig + OffPolicyConfig + SgmConfig + Clone,
    Env: Environment<Action = Act, Observation = Obs> + Renderable + 'static,
    Obs: Debug + Clone + Eq + Hash + TensorConvertible + DistanceMeasure + 'static,
    Act: Clone + TensorConvertible + Sampleable + 'static,
{
    pub fn open(
        env: Env,
        agent: Alg,
        config: Alg::Config,
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
            render_fancy: false,
        };
        eframe::run_native(
            "Actor-Critic Graph-Learner",
            eframe::NativeOptions {
                min_window_size: Some(egui::vec2(800.0, 600.0)),
                ..Default::default()
            },
            Box::new(|_| Box::new(gui)),
        )
        .unwrap();
    }

    fn run_gui_logic(&mut self) -> Result<()> {
        #[allow(clippy::collapsible_if)]
        // construct the graph every defined number of episodes
        if self.config.sgm_freq() > 0 && (self.run_data.len() + 1) % self.config.sgm_freq() == 0 {
            // but take care not to construct it constantly in this edge-case
            if self.last_graph_constructed_at != self.run_data.len() {
                self.graph = self
                    .agent
                    .replay_buffer()
                    .construct_sgm(
                        |o1, o2| <Obs>::distance(o1, o2),
                        self.config.sgm_maxdist(),
                        self.config.sgm_tau(),
                    )
                    .0;
                self.graph_egui = Graph::from(&self.graph);
                self.last_graph_constructed_at = self.run_data.len();
            }
        }

        // let it play to observe agent behavior!
        match self.play_mode {
            PlayMode::Pause => (),
            PlayMode::Ticks => {
                tick(&mut self.env, &mut self.agent, &self.device)?;
            }
            PlayMode::Episodes => {
                let mut config = self.config.clone();
                config.set_max_episodes(1);
                config.set_initial_random_actions(0);
                let (mc_returns, successes) = training_loop_off_policy(
                    &mut self.env,
                    &mut self.agent,
                    config,
                    &self.device,
                )?;
                self.run_data
                    .push((self.agent.run_mode(), mc_returns[0], successes[0]));
            }
        }
        Ok(())
    }

    fn run_agent(&mut self) -> Result<()> {
        let (mc_returns, successes) = training_loop_off_policy(
            &mut self.env,
            &mut self.agent,
            self.config.clone(),
            &self.device,
        )?;
        self.run_data.extend(
            (0..self.config.max_episodes())
                .map(|i| (RunMode::Train, mc_returns[i], successes[i])),
        );
        Ok(())
    }

    fn reset_agent(&mut self) -> Result<()> {
        let size_state = self.env.observation_space().iter().product::<usize>();
        let size_action = self.env.action_space().iter().product::<usize>();
        self.agent = *<Alg>::from_config(&self.device, &self.config, size_state, size_action)?;
        Ok(())
    }

    fn render_graph(
        &self,
        plot_ui: &mut PlotUi,
    ) {
        for edge in self.graph.edge_references() {
            let s1 = &self.graph[edge.source()];
            let s2 = &self.graph[edge.target()];

            let s1 = <Obs>::to_vec(s1.clone());
            let s2 = <Obs>::to_vec(s2.clone());

            plot_ui.line(
                Line::new(vec![[s1[0], s1[1]], [s2[0], s2[1]]])
                    .width(1.0)
                    .color(Color32::LIGHT_BLUE),
            )
        }
    }

    fn render_buffer(
        &self,
        plot_ui: &mut PlotUi,
    ) {
        for state in self.agent.replay_buffer().all_states::<Obs>() {
            let s = <Obs>::to_vec(state.clone());
            plot_ui.points(
                Points::new(vec![[s[0], s[1]]])
                    .radius(2.0)
                    .color(Color32::RED),
            );
        }
    }

    fn render_returns(
        &mut self,
        plot_ui: &mut PlotUi,
    ) {
        let (min, max) = self.env.value_range();
        let n = self.run_data.len() as f64;

        plot_ui.set_plot_bounds(PlotBounds::from_min_max([0.0, min], [n, max]));

        let _ = self
            .run_data
            .iter()
            .enumerate()
            .map(|(idx, (run_mode, mc_return, success))| {
                plot_ui.points(Points::new([idx as f64, *mc_return]).color(match run_mode {
                    RunMode::Train => Color32::RED,
                    RunMode::Test => Color32::GREEN,
                }));
                if *success {
                    plot_ui.line(
                        Line::new(vec![[idx as f64, min], [idx as f64, max]])
                            .width(1.0)
                            .color(Color32::LIGHT_GREEN),
                    )
                }
            })
            .collect::<Vec<_>>();
    }

    fn render_fancy(
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
        let mut max_episodes = self.config.max_episodes();
        let mut train_iterations = self.config.training_iterations();
        let mut init_random_actions = self.config.initial_random_actions();

        let mut actor_lr = self.config.actor_lr();
        let mut critic_lr = self.config.critic_lr();
        let mut gamma = self.config.gamma();
        let mut tau = self.config.tau();

        let mut buffer_size = self.config.replay_buffer_capacity();
        let mut batch_size = self.config.training_batch_size();

        let mut sgm_freq = self.config.sgm_freq();
        let mut sgm_maxdist = self.config.sgm_maxdist();
        let mut sgm_tau = self.config.sgm_tau();


        ui.separator();
        ui.label("DDPG Options");
        ui.add(
            Slider::new(&mut actor_lr, 0.00001..=0.1)
                .logarithmic(true)
                .fixed_decimals(5)
                .text("Actor LR"),
        );
        ui.add(
            Slider::new(&mut critic_lr, 0.00001..=0.1)
                .logarithmic(true)
                .fixed_decimals(5)
                .text("Critic LR"),
        );
        ui.add(
            Slider::new(&mut gamma, 0.0..=1.0)
                .step_by(0.01)
                .text("Gamma"),
        );
        ui.add(
            Slider::new(&mut tau, 0.001..=1.0)
                .logarithmic(true)
                .text("Tau"),
        );
        ui.add(
            Slider::new(&mut buffer_size, 10..=100_000)
                .logarithmic(true)
                .text("Buffer size"),
        );
        ui.add(
            Slider::new(&mut batch_size, 1..=1024)
                .step_by(1.0)
                .text("Batch size"),
        );
        ui.add(
            Slider::new(&mut train_iterations, 1..=200)
                .step_by(1.0)
                .text("Training iters"),
        );
        ui.add(
            Slider::new(&mut init_random_actions, 0..=1000)
                .text("Init. random actions"),
        );

        ui.separator();
        ui.label("SGM Options");
        ui.add(
            Slider::new(&mut sgm_freq, 0..=20)
                .step_by(1.0)
                .text("Rebuilding freq"),
        );
        ui.add(
            Slider::new(&mut sgm_maxdist, 0.0..=1.0)
                .step_by(0.01)
                .text("Max distance"),
        );
        ui.add(
            Slider::new(&mut sgm_tau, 0.0..=1.0)
                .step_by(0.01)
                .text("Tau"),
        );

        ui.separator();
        ui.label("Render Options");
        ui.add(Checkbox::new(&mut self.render_graph, "Show Graph"));
        ui.add(Checkbox::new(&mut self.render_buffer, "Show Buffer"));
        ui.add(Checkbox::new(&mut self.render_fancy, "Fancy GraphView"));

        ui.separator();
        ui.heading("Actions");
        ui.horizontal(|ui| {
            if ui.add(Button::new("Reset Agent")).clicked() {
                self.reset_agent().unwrap();
            };

            let agent_mode = match self.agent.run_mode() {
                RunMode::Test => "Test",
                RunMode::Train => "Train",
            };
            if ui
                .add(Button::new(format!("Toggle Mode ({agent_mode})")))
                .clicked()
            {
                self.agent.set_run_mode(match self.agent.run_mode() {
                    RunMode::Test => RunMode::Train,
                    RunMode::Train => RunMode::Test,
                });
            };
        });

        ui.separator();
        ui.label("Run Agent");
        ui.add(
            Slider::new(&mut max_episodes, 1..=1001)
                .step_by(1.0)
                .text("n_episodes"),
        );
        if ui.add(Button::new("Run Episodes")).clicked() {
            self.run_agent().unwrap();
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
    }
}
