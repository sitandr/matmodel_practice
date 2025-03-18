/// Test: math::exp(-(x/w - 0.5)^2 - (y/h - 0.5)^2)
/// max(1 - 10*(x/w - 0.5)^2 - 10*(y/h - 0.5)^2, 0)
/// if((cx/w)^2 + (y/h)^2 < 0.01, 1, 0)
/// if(nx == 2 && ny == 2, 1, 0)
/// if((cx/w)^2 + (y/h - 0.4)^2 < 0.004, 1, 0) + if((cx/w + 0.04)^2 + (0.6-y/h)^2 < 0.004, 1, 0)
/// 
/// interaction: 1/(f + 0.2) + 0.7

use std::time::Duration;
use evalexpr::EvalexprResult;

use crate::model::{EquationType, Modelling};

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct TemplateApp {
    model: Modelling,
    dx: f64,
    dt: f64,
    width: usize,
    height: usize,
    pause: bool,
    f_query: String,
    param_query: String,
    enable_param_f: bool,
    param_f_query: String,
    error: String,
    color_image: bool,
    steps_per_frame: usize,
    pause_interval: f64
    // #[serde(skip)] // This how you opt-out of serialization of a field
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
            dx: 0.5,
            dt: 0.05,
            width: 50,
            height: 50,
            pause: false,
            enable_param_f: false,
            param_f_query: "1.0".to_owned(),
            model: Modelling::init(50, 50),
            f_query: "0.0".to_owned(),
            error: Default::default(),
            param_query: "1.0".to_owned(),
            color_image: false,
            steps_per_frame: 1,
            pause_interval: 0.1
        }
    }
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        {
            if let Some(storage) = cc.storage {
                return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
            }

            Default::default()
        }
    }

    fn set_error(&mut self, err: EvalexprResult<()>) {
        if let Err(err) = err {
            self.error = err.to_string()
        } else {
            self.error = String::new();
        }
    }
}

impl eframe::App for TemplateApp {
    /// Called by the frame work to save state before shutdown.
    // fn save(&mut self, storage: &mut dyn eframe::Storage) {
        // eframe::set_value(storage, eframe::APP_KEY, self);
    // }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:

            egui::menu::bar(ui, |ui| {
                // NOTE: no File->Quit on web pages!
                let is_web = cfg!(target_arch = "wasm32");
                if !is_web {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.add_space(16.0);
                }

                egui::widgets::global_theme_preference_buttons(ui);
            });
        });

        egui::SidePanel::new(egui::panel::Side::Right, "Controls").show(ctx, |ui| {
            ui.separator();
            ui.add(egui::Slider::new(&mut self.model.flux.alpha, 0.0..=1.0).logarithmic(true).text("alpha"));
            ui.add(egui::Slider::new(&mut self.model.flux.beta, 0.0..=1.0).logarithmic(true).text("beta"));
            ui.add(egui::Slider::new(&mut self.model.flux.flux, 0.0..=1.0).logarithmic(true).text("flux"));
            ui.add(egui::Slider::new(&mut self.model.flux.up, -1.0..=1.0).step_by(0.5).text("top side flux multi"));
            ui.add(egui::Slider::new(&mut self.model.flux.down, -1.0..=1.0).step_by(0.5).text("bottom side flux multi"));
            ui.add(egui::Slider::new(&mut self.model.flux.left, -1.0..=1.0).step_by(0.5).text("left side flux multi"));
            ui.add(egui::Slider::new(&mut self.model.flux.right, -1.0..=1.0).step_by(0.5).text("right side flux multi"));

            ui.add(egui::Checkbox::new(&mut self.model.automax, "auto max"));
            ui.add(egui::Slider::new(&mut self.model.max, -100.0..=100.0).text("display max"));
            ui.add(egui::Slider::new(&mut self.model.min, -100.0..=100.0).text("display min"));

            ui.separator();

            egui::ComboBox::from_label("Equation type")
                .selected_text(format!("{:?}", self.model.eq_type))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.model.eq_type, EquationType::Wave, "Wave equation");
                    ui.selectable_value(&mut self.model.eq_type, EquationType::Heat, "Heat equation");
                    ui.selectable_value(&mut self.model.eq_type, EquationType::WaveInnerParam, "Wave equation with differentiated variable constant");
                    ui.selectable_value(&mut self.model.eq_type, EquationType::HeatInnerParam, "Heat equation with differentiated variable constant");
                }
            );
            
            ui.add(egui::Slider::new(&mut self.model.c0, 0.0..=10.0).text("equation constant c₀"));
            
            ui.horizontal(|ui| {
                ui.add(egui::Checkbox::new(&mut self.enable_param_f, "Enable c(f) dependency\n multiplier"));
                if self.enable_param_f {
                    ui.add(egui::TextEdit::singleline(&mut self.param_f_query));
                }
            });
            ui.horizontal(|ui| {
                ui.label("c(x)/c₀");
                ui.add(egui::TextEdit::singleline(&mut self.param_query));
                if ui.button("Apply").clicked() {
                    let err = self.model.set_param(&self.param_query, self.dx);
                    self.set_error(err);
                    let err = self.model.set_param_f(&self.param_f_query, self.enable_param_f);
                    self.set_error(err);
                }
            });
            
            ui.horizontal(|ui| {
                ui.label("Initial funtion");
                ui.add(egui::TextEdit::singleline(&mut self.f_query));
            });

            if ui.button("Fill by rule").clicked() {
                let err = self.model.set_f(&self.f_query, self.dx);
                self.set_error(err);
            }
            ui.add(egui::Label::new(&self.error));

            ui.add(egui::Slider::new(&mut self.dx, 0.0001..=1.0).logarithmic(true).text("dx"));
            ui.add(egui::Slider::new(&mut self.dt, 0.0001..=1.0).logarithmic(true).text("dt"));
            ui.add(egui::Slider::new(&mut self.width, 1..=500).text("width"));
            ui.add(egui::Slider::new(&mut self.height, 1..=500).text("height"));
            
            if ui.button("Reset & refill").clicked() {
                self.model.reset(self.width, self.height);
                self.model.dx = self.dx;
                self.model.dt = self.dt;
                let err = self.model.set_f(&self.f_query, self.dx);
                self.set_error(err);
                let err = self.model.set_param(&self.param_query, self.dx);
                self.set_error(err);
                let err = self.model.set_param_f(&self.param_f_query, self.enable_param_f);
                self.set_error(err);
            }

            ui.add(egui::Checkbox::new(&mut self.color_image, "Colored output"));
            ui.add(egui::Slider::new(&mut self.steps_per_frame, 1..=50).text("Steps per frame"));
            ui.add(egui::Slider::new(&mut self.pause_interval, 0.0..=1000.0).text("Sleep between frames, ms"));
            ui.add(egui::Checkbox::new(&mut self.pause, "Pause"));

            if ui.button("Save").clicked() {
                eframe::set_value(frame.storage_mut().unwrap(), eframe::APP_KEY, self);
            }


        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's
            ui.heading("Simple modelling");

            ui.separator();

            if !self.pause {
                let err = self.model.step(self.steps_per_frame, self.dx, self.dt);
                self.set_error(err);
                std::thread::sleep(Duration::from_secs_f64(self.pause_interval * 0.001));
                ui.ctx().request_repaint();
            }
            self.model.display(ui, self.color_image);
            if ui.add(egui::Button::new("Print")).clicked() {
                println!("{:?}", &self.model.data);
            }

            /*ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                powered_by_egui_and_eframe(ui);
                egui::warn_if_debug_build(ui);
            });*/
        });
    }
}

#[allow(dead_code)]
fn powered_by_egui_and_eframe(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.label("Powered by ");
        ui.hyperlink_to("egui", "https://github.com/emilk/egui");
        ui.label(" and ");
        ui.hyperlink_to(
            "eframe",
            "https://github.com/emilk/egui/tree/master/crates/eframe",
        );
        ui.label(".");
    });
}
