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
// use petgraph::Graph;
// use petgraph::visit::EdgeRef;

use eframe::egui;
use egui::widgets::Button;
use egui::plot::{Plot, Line};
use egui::{Ui, Slider};


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
    play_mode: PlayMode,
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
            play_mode: PlayMode::Pause,
        };
        eframe::run_native(
            "Actor-Critic Graph-Learner",
            eframe::NativeOptions::default(),
            Box::new(|_| Box::new(gui)),
        ).unwrap();
    }

    // fn plot_graph(
    //     graph: &Graph<PointState, OrderedFloat<f32>>,
    //     plot_ui: &mut PlotUi,
    // ) {
    //     for edge in graph.edge_references() {
    //         let s1 = graph[edge.source()];
    //         let s2 = graph[edge.target()];

    //         plot_ui.line(
    //             Line::new(
    //                 vec![
    //                     [s1.x(), s1.y()],
    //                     [s2.x(), s2.y()],
    //                 ]
    //             )
    //             .width(1.0)
    //             .color(Color32::LIGHT_BLUE)
    //         )
    //     }
    // }

    fn render_rewards(
        &mut self,
        ui: &mut Ui,
    ) {
        Plot::new("data_plot").show(ui, |plot_ui| {
            // plot_ui.set_plot_bounds(
            //     PlotBounds::from_min_max(
            //         [0.0, 0.0],
            //         [width as f64, height as f64],
            //     )
            // );
            plot_ui.line(
                Line::new(
                    self.episodic_returns.clone()
                    .into_iter()
                    .enumerate()
                    .map(|(x, y)| {
                        [x as f64, y as f64]
                    })
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
        ui.label("Run Options");
        ui.add(Slider::new(&mut self.config.max_episodes, 1..=101).text("n_episodes"));
        if ui.add(Button::new("Train Episodes")).clicked() {
            let episodic_returns = run(
                &mut self.env,
                &mut self.agent,
                self.config.clone(),
                true,
                &self.device,
            ).unwrap();
            self.episodic_returns.extend(episodic_returns);
        };
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
        // render the settings and options
        egui::SidePanel::right("settings").show(ctx, |ui| {
            self.render_options(ui);
        });

        // render episodic rewards / learning curve
        egui::TopBottomPanel::bottom("rewards").show(ctx, |ui| {
            self.render_rewards(ui);
        });

        // render the environment
        egui::CentralPanel::default().show(ctx, |ui| {
            self.env.render(ui);
        });

        // sleep for a bit
        thread::sleep(time::Duration::from_millis(100));

        // always repaint, not just on mouse-hover
        ctx.request_repaint();
    }
}
