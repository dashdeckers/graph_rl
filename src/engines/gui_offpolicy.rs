use {
    super::{
        tick_off_policy,
        loop_off_policy,
        ParamAlg,
        ParamEnv,
        ParamRunMode,
    },
    crate::{
        agents::{
            RunMode,
            Algorithm,
            OffPolicyAlgorithm,
        },
        envs::{
            Environment,
            RenderableEnvironment,
            Sampleable,
            TensorConvertible,
        },
        configs::{
            RenderableConfig,
            TrainConfig,
            TestConfig,
        },
    },
    anyhow::Result,
    serde::Serialize,
    candle_core::Device,
    eframe::egui,
    egui::{
        widgets::Button,
        Checkbox,
        Color32,
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
        thread,
        time,
        panic::{
            catch_unwind,
            AssertUnwindSafe,
        },
    },
};

pub enum PlayMode {
    Pause,
    Ticks,
    Episodes,
}

pub struct OffPolicyGUI<Alg, Env, Obs, Act>
where
    Env: Environment<Action = Act, Observation = Obs> + RenderableEnvironment,
    Env::Config: Clone + Serialize + RenderableConfig,
    Alg: Algorithm,
    Alg::Config: Clone + Serialize + RenderableConfig,
    Obs: Clone,
{
    pub env: Env,
    pub alg: Alg,
    pub env_config: Env::Config,
    pub alg_config: Alg::Config,
    pub run_mode: ParamRunMode,
    pub device: Device,

    pub run_data: Vec<(RunMode, f64, bool)>,
    pub play_mode: PlayMode,

    pub render_buffer: bool,
}

impl<Alg, Env, Obs, Act> eframe::App for OffPolicyGUI<Alg, Env, Obs, Act>
where
    Env: Clone + Environment<Action = Act, Observation = Obs> + RenderableEnvironment + 'static,
    Env::Config: Clone + Serialize + RenderableConfig,
    Alg: Clone + Algorithm + OffPolicyAlgorithm + 'static,
    Alg::Config: Clone + Serialize + RenderableConfig,
    Obs: Clone + TensorConvertible + 'static,
    Act: Clone + TensorConvertible + Sampleable + 'static,
{
    fn update(
        &mut self,
        ctx: &egui::Context,
        _frame: &mut eframe::Frame,
    ) {
        // render the settings and options
        egui::SidePanel::left("settings").show(ctx, |ui| {
            self.render_settings(ui);
            self.render_gui_options(ui);
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
    Env: Clone + Environment<Action = Act, Observation = Obs> + RenderableEnvironment + 'static,
    Env::Config: Clone + Serialize + RenderableConfig,
    Alg: Clone + Algorithm + OffPolicyAlgorithm + 'static,
    Alg::Config: Clone + Serialize + RenderableConfig,
    Obs: Clone + TensorConvertible + 'static,
    Act: Clone + TensorConvertible + Sampleable + 'static,
{
    pub fn create(
        init_env: ParamEnv<Env, Obs, Act>,
        init_alg: ParamAlg<Alg>,
        run_mode: ParamRunMode,
        device: Device,
    ) -> Self {
        let (env, env_config) = match init_env {
            ParamEnv::AsEnvironment(env) => (env.clone(), env.config().clone()),
            ParamEnv::AsConfig(config) => {
                let env = *Env::new(config.clone()).unwrap();
                (env.clone(), env.config().clone())
            },
        };

        let (alg, alg_config) = match &init_alg {
            ParamAlg::AsAlgorithm(alg) => (alg.clone(), alg.config().clone()),
            ParamAlg::AsConfig(config) => {
                let alg = *Alg::from_config(
                    &device,
                    config,
                    env.observation_space().iter().product::<usize>(),
                    env.action_space().iter().product::<usize>(),
                ).unwrap();
                (alg.clone(), alg.config().clone())
            },
        };

        Self {
            env,
            alg,
            env_config,
            alg_config,
            run_mode,
            device,

            run_data: Vec::new(),
            play_mode: PlayMode::Pause,

            render_buffer: false,
        }
    }

    pub fn open(
        init_env: ParamEnv<Env, Obs, Act>,
        init_alg: ParamAlg<Alg>,
        run_mode: ParamRunMode,
        device: Device,
    ) {
        let _ = catch_unwind(AssertUnwindSafe(|| eframe::run_native(
            "Actor-Critic Graph-Learner",
            eframe::NativeOptions {
                min_window_size: Some(egui::vec2(800.0 * 1.4, 600.0 * 1.4)),
                ..Default::default()
            },
            Box::new(|_| Box::new(Self::create(
                init_env,
                init_alg,
                run_mode,
                device,
            ))),
        )));
    }

    pub fn test_agent(&mut self) -> Result<()> {

        let mode = match &self.run_mode {
            ParamRunMode::Train(_) => RunMode::Train,
            ParamRunMode::Test(_) => RunMode::Test,
        };

        match self.play_mode {
            PlayMode::Pause => (),
            PlayMode::Ticks => {
                tick_off_policy(
                    &mut self.env,
                    &mut self.alg,
                    mode,
                    &self.device,
                )?;
            }
            PlayMode::Episodes => {
                let (mc_returns, successes) = loop_off_policy(
                    &mut self.env,
                    &mut self.alg,
                    match &self.run_mode {
                        ParamRunMode::Train(_) => ParamRunMode::Train(
                            TrainConfig::new(
                                1,
                                0,
                                0,
                            )
                        ),
                        ParamRunMode::Test(_) => ParamRunMode::Test(
                            TestConfig::new(1)
                        ),
                    },
                    &self.device,
                )?;
                self.run_data.push((mode, mc_returns[0], successes[0]));
            }
        }
        Ok(())
    }

    pub fn run_agent(&mut self) -> Result<()> {
        let (mc_returns, successes) = loop_off_policy(
            &mut self.env,
            &mut self.alg,
            self.run_mode.clone(),
            &self.device,
        )?;

        let (n, mode) = match &self.run_mode {
            ParamRunMode::Test(config) => (config.max_episodes(), RunMode::Test),
            ParamRunMode::Train(config) => (config.max_episodes(), RunMode::Train),
        };

        self.run_data.extend((0..n).map(|i| (mode, mc_returns[i], successes[i])));
        Ok(())
    }

    pub fn render_buffer(
        &self,
        plot_ui: &mut PlotUi,
    ) {
        for state in self.alg.replay_buffer().all_states::<Obs>() {
            let s = <Obs>::to_vec(state.clone());
            plot_ui.points(
                Points::new(vec![[s[0], s[1]]])
                    .radius(2.0)
                    .color(Color32::RED),
            );
        }
    }

    pub fn render_returns(
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

    pub fn render_settings(
        &mut self,
        ui: &mut Ui,
    ) {
        ui.heading("Settings");

        match self.run_mode.clone() {
            ParamRunMode::Test(mut config) => {
                config.render_mutable(ui);

                ui.horizontal(|ui| {
                    if ui
                        .add(Button::new("Run"))
                        .clicked()
                    {
                        self.run_agent().unwrap();
                        println!("Done!");
                    };
                    if ui
                        .add(Button::new("Toggle Mode"))
                        .clicked()
                    {
                        self.run_mode = ParamRunMode::Train(
                            TrainConfig::new(
                                config.max_episodes(),
                                30,
                                0,
                            ),
                        );
                    } else {
                        self.run_mode = ParamRunMode::Test(config);
                    }
                });
            }
            ParamRunMode::Train(mut config) => {
                config.render_mutable(ui);

                ui.horizontal(|ui| {
                    if ui
                        .add(Button::new("Run"))
                        .clicked()
                    {
                        self.run_agent().unwrap();
                        println!("Done!");
                    };
                    if ui
                        .add(Button::new("Toggle Mode"))
                        .clicked()
                    {
                        self.run_mode = ParamRunMode::Test(
                            TestConfig::new(
                                config.max_episodes(),
                            ),
                        );
                    } else {
                        self.run_mode = ParamRunMode::Train(config);
                    }
                });
            }
        }

        self.env_config.render_mutable(ui);

        ui.horizontal(|ui| {
            if ui.add(Button::new("Reset")).clicked() {
                self.env_config = self.env.config().clone();
            };
            if ui.add(Button::new("Set (new) Environment")).clicked() {
                self.env = *Env::new(self.env_config.clone()).unwrap();
            };
        });


        self.alg_config.render_mutable(ui);

        ui.horizontal(|ui| {
            if ui.add(Button::new("Reset")).clicked() {
                self.alg_config = self.alg.config().clone();
            };
            if ui.add(Button::new("Set (new) Agent")).clicked() {
                let size_state = self.env.observation_space().iter().product::<usize>();
                let size_action = self.env.action_space().iter().product::<usize>();
                self.alg = *Alg::from_config(
                    &self.device,
                    &self.alg_config,
                    size_state,
                    size_action,
                ).unwrap();
                self.run_data = Vec::new();
            };
        });

        ui.separator();
        let mode = match &self.run_mode {
            ParamRunMode::Train(_) => "TrainMode",
            ParamRunMode::Test(_) => "TestMode",
        };
        ui.label(format!("Watch Agent ({mode})"));
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
        self.test_agent().unwrap();
    }

    pub fn render_gui_options(
        &mut self,
        ui: &mut Ui,
    ) {
        ui.separator();
        ui.label("Render Options");
        ui.add(Checkbox::new(&mut self.render_buffer, "Show Buffer"));
    }
}


