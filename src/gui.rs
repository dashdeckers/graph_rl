use std::{thread, time};

use crate::PointEnv::{
    PointState,
    PointLine,
};

use tracing::{instrument, warn};
use anyhow::Result;

use eframe::egui;
use egui::widgets::plot::PlotUi;
use egui::plot::Plot;
use egui::Color32;
use egui::plot::PlotBounds;


pub struct PointEnvReplayer {
    width: usize,
    height: usize,
    start: PointState,
    goal: PointState,
    walls: Vec<PointLine>,
    histories: Vec<Vec<PointState>>,
}
impl PointEnvReplayer {
    pub fn new(
        width: usize,
        height: usize,
        start: PointState,
        goal: PointState,
        walls: Vec<PointLine>,
        histories: Vec<Vec<PointState>>,
    ) -> Self {
        Self {
            width,
            height,
            start,
            goal,
            walls,
            histories,
        }
    }

    pub fn plot(
        &self,
        plot_ui: &mut PlotUi,
    ) {
        Self::setup_plot(self.width, self.height, plot_ui);
        Self::plot_walls(&self.walls, plot_ui);
        Self::plot_start_and_goal(&self.start, &self.goal, plot_ui);
        for history in self.histories.iter() {
            Self::plot_path(history, plot_ui);
        }
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
                        [wall.A.x() as f64, wall.A.y() as f64],
                        [wall.B.x() as f64, wall.B.y() as f64],
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
                    [p.x() as f64, p.y() as f64]
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
                    [start.x() as f64, start.y() as f64],
                ]
            )
            .radius(2.0)
            .color(Color32::WHITE)
        );
        plot_ui.points(
            egui::plot::Points::new(
                vec![
                    [goal.x() as f64, goal.y() as f64],
                ]
            )
            .radius(2.0)
            .color(Color32::GREEN)
        );
    }
}


#[instrument(skip(player))]
pub fn run_gui(
    player: PointEnvReplayer,
) -> Result<()> {
    warn!("Running GUI");
    eframe::run_native(
        "Actor-Critic Graph-Learner",
        eframe::NativeOptions::default(),
        Box::new(|_| Box::new(player)),
    ).unwrap();
    Ok(())
}


impl eframe::App for PointEnvReplayer {
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

