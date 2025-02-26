use {
    super::{
        ParamAlg,
        ParamEnv,
        OffPolicyGUI,
    },
    crate::{
        agents::{
            Algorithm,
            OffPolicyAlgorithm,
            HgbAlgorithm,
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
    },
    serde::Serialize,
    candle_core::Device,
    petgraph::visit::{
        EdgeRef,
        IntoEdgeReferences,
    },
    eframe::egui,
    egui::{
        Button,
        Checkbox,
        Color32,
        Ui,
    },
    egui_plot::{
        Line,
        Plot,
        PlotUi,
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

pub struct HgbGUI<Alg, Env, Obs, Act>
where
    Env: Environment<Action = Act, Observation = Obs> + RenderableEnvironment,
    Env::Config: Clone + Serialize + RenderableConfig,
    Alg: Algorithm,
    Alg::Config: Clone + Serialize + RenderableConfig,
    Obs: Clone,
{
    gui: OffPolicyGUI<Alg, Env, Obs, Act>,

    render_graph: bool,
    render_plan: bool,
}

impl<Alg, Env, Obs, Act> eframe::App for HgbGUI<Alg, Env, Obs, Act>
where
    Env: Clone + Environment<Action = Act, Observation = Obs> + RenderableEnvironment + 'static,
    Env::Config: Clone + Serialize + RenderableConfig,
    Alg: Clone + Algorithm + OffPolicyAlgorithm + SaveableAlgorithm + HgbAlgorithm<Env> + 'static,
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
                self.gui.render_settings(ui);
                if ui.add(Button::new("Clear Graph")).clicked() {
                    self.gui.alg.clear_graph();
                }
                if ui.add(Button::new("Reconstruct Graph")).clicked() {
                    self.gui.alg.construct_graph();
                }
                self.render_gui_options(ui);
            });
        });

        // render episodic rewards / learning curve
        egui::TopBottomPanel::top("rewards").show(ctx, |ui| {
            Plot::new("rewards_plot").show_axes([false; 2]).show(ui, |plot_ui| {
                self.gui.render_returns(plot_ui);
            });
        });

        // render the environment / graph
        egui::CentralPanel::default().show(ctx, |ui| {
            Plot::new("environment").show_axes([false; 2]).show(ui, |plot_ui| {
                //.view_aspect(1.0)
                self.gui.env.render(plot_ui);
                if self.render_graph {
                    self.render_graph(plot_ui);
                }
                if self.render_plan {
                    self.render_plan(plot_ui);
                }
                if self.gui.render_buffer {
                    self.gui.render_buffer(plot_ui);
                }
            });
        });

        // sleep for a bit
        thread::sleep(time::Duration::from_millis(100));

        // always repaint, not just on mouse-hover
        ctx.request_repaint();
    }
}

impl<Alg, Env, Obs, Act> HgbGUI<Alg, Env, Obs, Act>
where
    Env: Clone + Environment<Action = Act, Observation = Obs> + RenderableEnvironment + 'static,
    Env::Config: Clone + Serialize + RenderableConfig,
    Alg: Clone + Algorithm + OffPolicyAlgorithm + SaveableAlgorithm + HgbAlgorithm<Env> + 'static,
    Alg::Config: Clone + Serialize + RenderableConfig,
    Obs: Clone + TensorConvertible + 'static,
    Act: Clone + TensorConvertible + Sampleable + 'static,
{
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
            Box::new(|_| Box::new(Self {
                gui: OffPolicyGUI::<Alg, Env, Obs, Act>::create(
                    init_env,
                    init_alg,
                    config,
                    load_model,
                    pretrain_train_config,
                    pretrain_env_config,
                    device,
                ),
                render_graph: false,
                render_plan: false,
            })),
        )));
    }

    pub fn render_graph(
        &self,
        plot_ui: &mut PlotUi,
    ) {
        let graph = self.gui.alg.graph();
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

    pub fn render_plan(
        &self,
        plot_ui: &mut PlotUi,
    ) {
        let series = self.gui.alg
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

    pub fn render_gui_options(
        &mut self,
        ui: &mut Ui,
    ) {
        ui.separator();
        ui.label("Render Options");
        ui.add(Checkbox::new(&mut self.render_graph, "Show Graph"));
        ui.add(Checkbox::new(&mut self.render_plan, "Show Plan"));
        ui.add(Checkbox::new(&mut self.gui.render_buffer, "Show Buffer"));
    }
}


