use {
    super::{
        tick_off_policy,
        loop_off_policy,
        ParamAlg,
        ParamEnv,
    },
    crate::{
        agents::{
            Algorithm,
            OffPolicyAlgorithm,
            SaveableAlgorithm,
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
        },
        engines::RunMode,
    },
    anyhow::Result,
    tracing::warn,
    serde::Serialize,
    candle_core::Device,
    eframe::egui,
    egui::{
        widgets::{
            Button,
            Slider,
        },
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
        path::Path,
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
    pub config: TrainConfig,
    pub device: Device,

    pub run_data: Vec<(RunMode, f64, bool)>,
    pub play_mode: PlayMode,

    pub slowdown_ms: u64,
    pub slowdown_ticker: u64,

    pub render_buffer: bool,
}

impl<Alg, Env, Obs, Act> eframe::App for OffPolicyGUI<Alg, Env, Obs, Act>
where
    Env: Clone + Environment<Action = Act, Observation = Obs> + RenderableEnvironment + 'static,
    Env::Config: Clone + Serialize + RenderableConfig,
    Alg: Clone + Algorithm + OffPolicyAlgorithm + SaveableAlgorithm + 'static,
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
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.render_settings(ui);
                self.render_gui_options(ui);
            });
        });

        // render episodic rewards / learning curve
        egui::TopBottomPanel::top("rewards").show(ctx, |ui| {
            Plot::new("rewards_plot").show_axes([false; 2]).show(ui, |plot_ui| {
                self.render_returns(plot_ui);
            });
        });

        // render the environment / graph
        egui::CentralPanel::default().show(ctx, |ui| {
            Plot::new("environment").show_axes([false; 2]).show(ui, |plot_ui| {
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
    Alg: Clone + Algorithm + OffPolicyAlgorithm + SaveableAlgorithm + 'static,
    Alg::Config: Clone + Serialize + RenderableConfig,
    Obs: Clone + TensorConvertible + 'static,
    Act: Clone + TensorConvertible + Sampleable + 'static,
{
    pub fn create(
        init_env: ParamEnv<Env, Obs, Act>,
        init_alg: ParamAlg<Alg>,
        config: TrainConfig,
        load_model: Option<(String, String)>,
        pretrain_train_config: Option<TrainConfig>,
        pretrain_env_config: Option<Env::Config>,
        device: Device,
    ) -> Self {
        let (env, env_config) = match init_env {
            ParamEnv::AsEnvironment(env) => (env.clone(), env.config().clone()),
            ParamEnv::AsConfig(config) => {
                let env = *Env::new(config.clone()).unwrap();
                (env.clone(), env.config().clone())
            },
        };

        let (mut alg, alg_config) = match &init_alg {
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

        // Maybe load model weights

        if let Some((model_path, model_name)) = load_model {
            warn!("Loading model weights from {model_path} with name {model_name}");
            alg.load(
                &Path::new(&model_path),
                &model_name,
            ).unwrap();
        }


        // Maybe pretrain the Agent

        if let Some(pretrain_train_config) = pretrain_train_config.clone() {

            let (pretrain_mc_returns, _) = loop_off_policy(
                &mut match pretrain_env_config {
                    Some(ref env_config) => *Env::new(env_config.clone()).unwrap(),
                    None => env.clone(),
                },
                &mut alg,
                pretrain_train_config,
                &device,
            ).unwrap();

            warn!(
                "Pretrained with Avg return: \n{:#?}",
                pretrain_mc_returns.iter().sum::<f64>() / pretrain_mc_returns.len() as f64,
            );

            warn!(
                "Size of Replay Buffer: {:#?}",
                alg.replay_buffer().size(),
            )
        }

        Self {
            env,
            alg,
            env_config,
            alg_config,
            config,
            device,

            run_data: Vec::new(),
            play_mode: PlayMode::Pause,

            slowdown_ms: 0,
            slowdown_ticker: 0,

            render_buffer: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open(
        init_env: ParamEnv<Env, Obs, Act>,
        init_alg: ParamAlg<Alg>,
        config: TrainConfig,
        load_model: Option<(String, String)>,
        pretrain_train_config: Option<TrainConfig>,
        pretrain_env_config: Option<Env::Config>,
        device: Device,
        size: f32,
    ) {
        let _ = catch_unwind(AssertUnwindSafe(|| eframe::run_native(
            "Actor-Critic Graph-Learner",
            eframe::NativeOptions {
                min_window_size: Some(egui::vec2(800.0 * size, 600.0 * size)),
                ..Default::default()
            },
            Box::new(|_| Box::new(Self::create(
                init_env,
                init_alg,
                config,
                load_model,
                pretrain_train_config,
                pretrain_env_config,
                device,
            ))),
        )));
    }

    pub fn test_agent(&mut self) -> Result<()> {
        if self.slowdown_ticker > 0 {
            thread::sleep(time::Duration::from_millis(10));
            self.slowdown_ticker -= 10;
        } else {
            self.slowdown_ticker = self.slowdown_ms;

            match self.play_mode {
                PlayMode::Pause => (),
                PlayMode::Ticks => {
                    tick_off_policy(
                        &mut self.env,
                        &mut self.alg,
                        self.config.run_mode(),
                        &self.device,
                    )?;
                }
                PlayMode::Episodes => {
                    let (mc_returns, successes) = loop_off_policy(
                        &mut self.env,
                        &mut self.alg,
                        TrainConfig::new(
                            1,
                            0,
                            0,
                            self.config.run_mode(),
                        ),
                        &self.device,
                    )?;
                    self.run_data.push((self.config.run_mode(), mc_returns[0], successes[0]));
                }
            }
        }
        Ok(())
    }

    pub fn run_agent(&mut self) -> Result<()> {
        let (mc_returns, successes) = loop_off_policy(
            &mut self.env,
            &mut self.alg,
            self.config.clone(),
            &self.device,
        )?;

        self.run_data.extend((0..self.config.max_episodes()).map(|i| (self.config.run_mode(), mc_returns[i], successes[i])));
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

        self.config.render_mutable(ui);

        ui.horizontal(|ui| {
            if ui
                .add(Button::new("Run"))
                .clicked()
            {
                self.run_agent().unwrap();
                println!("Done!");
            };
        });

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
            if ui.add(Button::new("Override Agent")).clicked() {
                self.alg.override_config(&self.alg_config);
                self.alg_config = self.alg.config().clone();
            };
        });
        if ui.add(Button::new("Save Agent")).clicked() {
            self.alg.save(
                &Path::new("data/"),
                "GUI-saved",
            ).unwrap();
        };

        ui.separator();
        let mode = self.config.run_mode();
        ui.label(format!("Watch Agent ({mode:?})"));
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
        ui.add(
            Slider::new(&mut self.slowdown_ms, 0..=100)
                .step_by(10.0)
                .text("Tick Slowdown"),
        );
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


