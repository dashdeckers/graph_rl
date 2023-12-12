use {
    super::{
        tick,
        tick_off_policy,
        training_loop_off_policy,
    },
    crate::{
        agents::{
            Algorithm,
            OffPolicyAlgorithm,
            SgmAlgorithm,
            configs::{
                AlgorithmConfig,
                ActorCriticConfig,
                OffPolicyConfig,
                SgmConfig,
                DistanceMode,
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
        Label,
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
    TicksStatic,
    TicksPlastic,
    Episodes,
}

pub struct SgmGUI<Alg, Env, Obs, Act>
where
    Env: Environment<Action = Act, Observation = Obs> + Renderable,
    Alg: Algorithm,
    Alg::Config: AlgorithmConfig,
    Obs: Clone,
{
    env: Env,
    agent: Alg,
    config: Alg::Config,
    device: Device,

    run_data: Vec<(RunMode, f64, bool)>,
    play_mode: PlayMode,
    egui_graph: Graph<Obs, OrderedFloat<f64>, Undirected>,

    render_graph: bool,
    render_plan: bool,
    render_buffer: bool,
    render_fancy: bool,

    tick_slowdown: u64,
}

impl<Alg, Env, Obs, Act> eframe::App for SgmGUI<Alg, Env, Obs, Act>
where
    Env: Environment<Action = Act, Observation = Obs> + Renderable + 'static,
    Env::Config: Clone,
    Alg: Algorithm + OffPolicyAlgorithm + SgmAlgorithm<Env> + 'static,
    Alg::Config: Clone + AlgorithmConfig + ActorCriticConfig + OffPolicyConfig + SgmConfig,
    Obs: Clone + Debug + Eq + Hash + TensorConvertible + DistanceMeasure + 'static,
    Act: Clone + TensorConvertible + Sampleable + 'static,
{
    fn update(
        &mut self,
        ctx: &egui::Context,
        _frame: &mut eframe::Frame,
    ) {
        // run the SgmGUI logic
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
                .egui_graph
                .g
                .node_indices()
                .find(|e| self.egui_graph.g[*e].selected());

            if let Some(node) = node {
                ui.label(format!(
                    "Selected node: {:#?}",
                    self.egui_graph.g[node].data()
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
                    if self.render_plan {
                        self.render_plan(plot_ui);
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

impl<Alg, Env, Obs, Act> SgmGUI<Alg, Env, Obs, Act>
where
    Env: Environment<Action = Act, Observation = Obs> + Renderable + 'static,
    Env::Config: Clone,
    Alg: Algorithm + OffPolicyAlgorithm + SgmAlgorithm<Env> + 'static,
    Alg::Config: Clone + AlgorithmConfig + ActorCriticConfig + OffPolicyConfig + SgmConfig,
    Obs: Clone + Debug + Eq + Hash + TensorConvertible + DistanceMeasure + 'static,
    Act: Clone + TensorConvertible + Sampleable + 'static,
{
    pub fn open(
        env_config: Env::Config,
        alg_config: Alg::Config,
        agent: Option<Alg>,
        device: Device,
    ) {
        let env = *Env::new(env_config.clone()).unwrap();
        let agent = agent.unwrap_or(*Alg::from_config(
            &device,
            &alg_config,
            env.observation_space().iter().product::<usize>(),
            env.action_space().iter().product::<usize>(),
        ).unwrap());

        let gui = Self {
            env,
            agent,
            config: alg_config,
            device,

            run_data: Vec::new(),
            play_mode: PlayMode::Pause,
            egui_graph: Graph::from(&StableGraph::default()),

            render_graph: false,
            render_plan: false,
            render_buffer: false,
            render_fancy: false,

            tick_slowdown: 0,
        };
        eframe::run_native(
            "Actor-Critic Graph-Learner",
            eframe::NativeOptions {
                min_window_size: Some(egui::vec2(800.0 * 1.2, 600.0 * 1.2)),
                ..Default::default()
            },
            Box::new(|_| Box::new(gui)),
        )
        .unwrap();
    }

    fn run_gui_logic(&mut self) -> Result<()> {
        // let it play to observe agent behavior!
        match self.play_mode {
            PlayMode::Pause => (),
            PlayMode::TicksStatic => {
                tick(&mut self.env, &mut self.agent, &self.device)?;
                thread::sleep(time::Duration::from_millis(self.tick_slowdown));
            }
            PlayMode::TicksPlastic => {
                tick_off_policy(&mut self.env, &mut self.agent, &self.device)?;
                thread::sleep(time::Duration::from_millis(self.tick_slowdown));
            }
            PlayMode::Episodes => {
                let mut config = self.config.clone();
                config.set_max_episodes(1);
                config.set_initial_random_actions(0);
                config.set_training_iterations(0);
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
        let graph = self.agent.graph();
        for edge in graph.edge_references() {
            let o1 = &graph[edge.source()];
            let o2 = &graph[edge.target()];

            let s1 = <Obs>::to_vec(o1.clone());
            let s2 = <Obs>::to_vec(o2.clone());

            plot_ui.line(
                Line::new(vec![[s1[0], s1[1]], [s2[0], s2[1]]])
                    .width(1.0)
                    .color(Color32::LIGHT_BLUE),
            )
        }
    }

    fn render_plan(
        &self,
        plot_ui: &mut PlotUi,
    ) {
        let series = self.agent
            .plan()
            .iter()
            .map(|o| {
                let obs = <Obs>::to_vec(o.clone());
                [obs[0], obs[1]]
            })
            .collect::<Vec<[f64; 2]>>();

        plot_ui.line(
            Line::new(series)
                .width(1.0)
                .color(Color32::YELLOW),
        )
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
        // construct the egui graph
        self.egui_graph = Graph::from(self.agent.graph());

        let interaction_settings = &SettingsInteraction::new()
            .with_dragging_enabled(true)
            .with_clicking_enabled(true)
            .with_selection_enabled(true);
        let style_settings = &SettingsStyle::new().with_labels_always(true);
        ui.add(
            &mut GraphView::new(&mut self.egui_graph)
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

        let actor_lr = self.config.actor_lr();
        let critic_lr = self.config.critic_lr();
        let gamma = self.config.gamma();
        let tau = self.config.tau();

        let buffer_size = self.config.replay_buffer_capacity();
        let batch_size = self.config.training_batch_size();

        let mut sgm_close_enough = self.config.sgm_close_enough();
        let mut sgm_maxdist = self.config.sgm_maxdist();
        let mut sgm_tau = self.config.sgm_tau();

        ui.separator();
        ui.label("DDPG Options");
        ui.add(Label::new(format!("Actor LR: {actor_lr:#.5}")));
        ui.add(Label::new(format!("Critic LR: {critic_lr:#.5}")));
        ui.add(Label::new(format!("Gamma: {gamma}")));
        ui.add(Label::new(format!("Tau: {tau}")));
        ui.add(Label::new(format!("Buffer size: {buffer_size}")));
        ui.add(Label::new(format!("Batch size: {batch_size}")));
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
            Slider::new(&mut sgm_close_enough, 0.0..=2.0)
                .step_by(0.1)
                .text("Close enough"),
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
        let sgm_dist_mode = self.config.sgm_dist_mode();
        if ui
            .add(Button::new(format!("Toggle DistMode ({sgm_dist_mode})")))
            .clicked()
        {
            self.config.set_sgm_dist_mode(match self.config.sgm_dist_mode() {
                DistanceMode::True => DistanceMode::Estimated,
                DistanceMode::Estimated => DistanceMode::True,
            });
        };
        if ui.add(Button::new("Construct Graph")).clicked() {
            self.agent.construct_graph();
        };

        ui.separator();
        ui.label("Render Options");
        ui.add(Checkbox::new(&mut self.render_graph, "Show Graph"));
        ui.add(Checkbox::new(&mut self.render_plan, "Show Plan"));
        ui.add(Checkbox::new(&mut self.render_buffer, "Show Buffer"));
        ui.add(Checkbox::new(&mut self.render_fancy, "Fancy GraphView"));

        ui.separator();
        ui.heading("Actions");
        ui.horizontal(|ui| {
            if ui.add(Button::new("Reset Agent")).clicked() {
                self.reset_agent().unwrap();
            };

            let agent_mode = self.agent.run_mode();
            if ui
                .add(Button::new(format!("Toggle TrainMode ({agent_mode})")))
                .clicked()
            {
                self.agent.set_run_mode(match self.agent.run_mode() {
                    RunMode::Test => RunMode::Train,
                    RunMode::Train => RunMode::Test,
                });
            };
        });

        ui.separator();
        ui.label("Train Agent");
        ui.add(
            Slider::new(&mut max_episodes, 1..=501)
                .step_by(1.0)
                .text("n_episodes"),
        );
        if ui.add(Button::new("Run Episodes")).clicked() {
            self.run_agent().unwrap();
        };
        if ui.add(Button::new("Train only")).clicked() {
            for _ in 0..self.config.training_iterations() {
                self.agent.train().unwrap();
            }
        };

        ui.separator();
        ui.label("Test Agent");
        ui.horizontal(|ui| {
            if ui.add(Button::new("Pause")).clicked() {
                self.play_mode = PlayMode::Pause;
            };
            if ui.add(Button::new("Play(ts)")).clicked() {
                self.play_mode = PlayMode::TicksStatic;
            };
            if ui.add(Button::new("Play(tp)")).clicked() {
                self.play_mode = PlayMode::TicksPlastic;
            };
            if ui.add(Button::new("Play(e)")).clicked() {
                self.play_mode = PlayMode::Episodes;
            };
        });
        ui.add(
            Slider::new(&mut self.tick_slowdown, 0..=501)
                .step_by(1.0)
                .text("Set Tick Slowdown"),
        );

        self.config.set_max_episodes(max_episodes);
        self.config.set_training_iterations(train_iterations);
        self.config.set_initial_random_actions(init_random_actions);

        self.config.set_sgm_close_enough(sgm_close_enough);
        self.config.set_sgm_maxdist(sgm_maxdist);
        self.config.set_sgm_tau(sgm_tau);

        self.agent.set_from_config(&self.config);
    }
}
