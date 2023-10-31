use std::{thread, time};

use crate::{
    ddpg::DDPG,
    envs::point_env::{
        PointEnv,
        PointState,
        PointLine,
    },
};

use eframe::egui;
use egui::widgets::plot::PlotUi;
use egui::plot::Plot;
use egui::Color32;
use egui::plot::PlotBounds;



pub struct GUI<'a> {
    env: PointEnv,
    #[allow(dead_code)]
    agent: DDPG<'a>,
}
impl GUI<'static> {
    pub fn new(
        env: PointEnv,
        agent: DDPG<'static>,
    ) -> Self {
        Self { env, agent }
    }

    pub fn show(gui: Self) {
        eframe::run_native(
            "Actor-Critic Graph-Learner",
            eframe::NativeOptions::default(),
            Box::new(|_| Box::new(gui)),
        ).unwrap();
    }

    pub fn plot(
        &self,
        plot_ui: &mut PlotUi,
    ) {
        Self::setup_plot(*self.env.width(), *self.env.height(), plot_ui);
        Self::plot_walls(self.env.walls(), plot_ui);
        Self::plot_start_and_goal(self.env.start(), self.env.goal(), plot_ui);
        Self::plot_path(self.env.history(), plot_ui);
    }

    fn setup_plot(
        width: usize,
        height: usize,
        plot_ui: &mut PlotUi,
    ) {
        plot_ui.set_plot_bounds(
            PlotBounds::from_min_max(
                [0.0, 0.0],
                [width as f64, height as f64],
            )
        );
    }

    fn plot_walls(
        walls: &[PointLine],
        plot_ui: &mut PlotUi,
    ) {
        for wall in walls.iter() {
            plot_ui.line(
                egui::plot::Line::new(
                    vec![
                        [wall.A.x(), wall.A.y()],
                        [wall.B.x(), wall.B.y()],
                    ]
                )
                .width(2.0)
                .color(Color32::WHITE)
            )
        }
    }

    fn plot_path(
        history: &[PointState],
        plot_ui: &mut PlotUi,
    ) {
        plot_ui.line(
            egui::plot::Line::new(
                history
                .iter()
                .map(|p| {
                    [p.x(), p.y()]
                })
                .collect::<Vec<_>>()
            )
        )
    }

    fn plot_start_and_goal(
        start: &PointState,
        goal: &PointState,
        plot_ui: &mut PlotUi,
    ) {
        plot_ui.points(
            egui::plot::Points::new(
                vec![
                    [start.x(), start.y()],
                ]
            )
            .radius(2.0)
            .color(Color32::WHITE)
        );
        plot_ui.points(
            egui::plot::Points::new(
                vec![
                    [goal.x(), goal.y()],
                ]
            )
            .radius(2.0)
            .color(Color32::GREEN)
        );
    }
}
impl eframe::App for GUI<'static> {
    fn update(
        &mut self,
        ctx: &egui::Context,
        _frame: &mut eframe::Frame,
    ) {
        // render the gui
        egui::CentralPanel::default().show(ctx, |ui| {
            // ui.heading("Central Panel");
            Plot::new("environment_plot").show(ui, |plot_ui| {
                self.plot(plot_ui)
            });
        });

        // sleep for a bit
        thread::sleep(time::Duration::from_millis(100));

        // always repaint, not just on mouse-hover
        ctx.request_repaint();
    }
}


// #[instrument(skip(player))]
// pub fn run_gui(
//     player: GUI<'static>,
// ) -> Result<()> {
//     warn!("Running GUI");
//     eframe::run_native(
//         "Actor-Critic Graph-Learner",
//         eframe::NativeOptions::default(),
//         Box::new(|_| Box::new(player)),
//     ).unwrap();
//     Ok(())
// }
