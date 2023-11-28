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
            },
        },
        envs::{
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
    egui_plot::{
        Line,
        Plot,
        PlotBounds,
        PlotUi,
        Points,
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

pub struct OffPolicyGUI<Alg, Env, Obs, Act>
where
    Alg: Algorithm,
    Alg::Config: AlgorithmConfig,
    Env: Environment<Action = Act, Observation = Obs> + Renderable,
    Obs: Clone,
{
    env: Env,
    agent: Alg,
    config: Alg::Config,
    device: Device,

    run_data: Vec<(RunMode, f64, bool)>,
    play_mode: PlayMode,

    render_buffer: bool,
}

impl<Alg, Env, Obs, Act> eframe::App for OffPolicyGUI<Alg, Env, Obs, Act>
where
    Alg: Algorithm + OffPolicyAlgorithm + 'static,
    Alg::Config: Clone + AlgorithmConfig + ActorCriticConfig + OffPolicyConfig,
    Env: Environment<Action = Act, Observation = Obs> + Renderable + 'static,
    Obs: Clone + Debug + Eq + Hash + TensorConvertible + 'static,
    Act: Clone + TensorConvertible + Sampleable + 'static,
{
    fn update(
        &mut self,
        ctx: &egui::Context,
        _frame: &mut eframe::Frame,
    ) {
        // run the OffPolicyGUI logic
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

        // render the environment / graph
        egui::CentralPanel::default().show(ctx, |ui| {
            Plot::new("environment").show(ui, |plot_ui| {
                //.view_aspect(1.0)
                self.env.render(plot_ui);
                if self.render_buffer {
                    self.render_buffer(plot_ui);
                }
            });
        });

        // sleep for a bit
        thread::sleep(time::Duration::from_millis(100));

        // always repaint, not just on mouse-hover
        ctx.request_repaint();
    }
}

impl<Alg, Env, Obs, Act> OffPolicyGUI<Alg, Env, Obs, Act>
where
    Alg: Algorithm + OffPolicyAlgorithm + 'static,
    Alg::Config: Clone + AlgorithmConfig + ActorCriticConfig + OffPolicyConfig,
    Env: Environment<Action = Act, Observation = Obs> + Renderable + 'static,
    Obs: Clone + Debug + Eq + Hash + TensorConvertible + 'static,
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
            play_mode: PlayMode::Pause,

            render_buffer: false,
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
        ui.label("Render Options");
        ui.add(Checkbox::new(&mut self.render_buffer, "Show Buffer"));

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
            Slider::new(&mut max_episodes, 1..=501)
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

        self.config.set_max_episodes(max_episodes);
        self.config.set_training_iterations(train_iterations);
        self.config.set_initial_random_actions(init_random_actions);

        self.config.set_actor_lr(actor_lr);
        self.config.set_critic_lr(critic_lr);
        self.config.set_gamma(gamma);
        self.config.set_tau(tau);

        self.config.set_replay_buffer_capacity(buffer_size);
        self.config.set_training_batch_size(batch_size);
    }
}
