//! Qompas — Quantum Exploration Tool
//!
//! Launches an interactive window (blade-graphics + egui) for building,
//! stepping through, and visualizing quantum circuits.

use qompas::viz::app::QompasApp;

fn main() {
    env_logger::init();

    // When no GPU / display is available, fall back to a CLI demo.
    if std::env::var("QOMPAS_HEADLESS").is_ok() || !has_display() {
        cli_demo();
        return;
    }

    run_windowed();
}

/// Detect whether a display server is available.
fn has_display() -> bool {
    std::env::var("WAYLAND_DISPLAY").is_ok() || std::env::var("DISPLAY").is_ok()
}

/// Launch the interactive GPU-accelerated visualizer.
fn run_windowed() {
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::EventLoop;
    use winit::window::Window;

    struct Harness {
        window: Option<Window>,
        gpu: Option<blade_graphics::Context>,
        surface: Option<blade_graphics::Surface>,
        gui_painter: Option<blade_egui::GuiPainter>,
        egui_ctx: egui::Context,
        app: QompasApp,
        window_size: (u32, u32),
    }

    impl ApplicationHandler for Harness {
        fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
            if self.window.is_some() {
                return; // already initialised
            }

            let attrs = Window::default_attributes()
                .with_title("Qompas — Quantum Explorer")
                .with_inner_size(winit::dpi::LogicalSize::new(1200u32, 700u32));
            let window = event_loop.create_window(attrs).expect("create window");

            let ctx_desc = blade_graphics::ContextDesc {
                presentation: true,
                validation: cfg!(debug_assertions),
                ..Default::default()
            };
            let gpu = unsafe { blade_graphics::Context::init(ctx_desc) }
                .expect("init GPU context");

            let size = window.inner_size();
            self.window_size = (size.width, size.height);

            let surface_config = blade_graphics::SurfaceConfig {
                size: blade_graphics::Extent {
                    width: size.width,
                    height: size.height,
                    depth: 1,
                },
                usage: blade_graphics::TextureUsage::TARGET,
                display_sync: blade_graphics::DisplaySync::Block,
                ..Default::default()
            };
            let surface = gpu
                .create_surface_configured(&window, surface_config)
                .expect("create surface");

            let surface_info = surface.info();
            let gui_painter = blade_egui::GuiPainter::new(surface_info, &gpu);

            log::info!(
                "Qompas launched. GPU: {}",
                gpu.device_information().device_name
            );

            self.window = Some(window);
            self.gpu = Some(gpu);
            self.surface = Some(surface);
            self.gui_painter = Some(gui_painter);
        }

        fn window_event(
            &mut self,
            event_loop: &winit::event_loop::ActiveEventLoop,
            _window_id: winit::window::WindowId,
            event: WindowEvent,
        ) {
            let gpu = self.gpu.as_ref().unwrap();
            let surface = self.surface.as_mut().unwrap();
            let gui_painter = self.gui_painter.as_mut().unwrap();

            match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::Resized(size) => {
                    self.window_size = (size.width, size.height);
                    let config = blade_graphics::SurfaceConfig {
                        size: blade_graphics::Extent {
                            width: size.width,
                            height: size.height,
                            depth: 1,
                        },
                        usage: blade_graphics::TextureUsage::TARGET,
                        display_sync: blade_graphics::DisplaySync::Block,
                        ..Default::default()
                    };
                    gpu.reconfigure_surface(surface, config);
                }
                WindowEvent::RedrawRequested => {
                    // Run egui frame
                    let raw_input = egui::RawInput::default();
                    let full_output = self.egui_ctx.run(raw_input, |ctx| {
                        self.app.ui(ctx);
                    });
                    let paint_jobs = self
                        .egui_ctx
                        .tessellate(full_output.shapes, self.egui_ctx.pixels_per_point());

                    // Render
                    let frame = surface.acquire_frame();
                    let mut encoder =
                        gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
                            name: "main",
                            buffer_count: 2,
                        });
                    encoder.start();

                    gui_painter.update_textures(
                        &mut encoder,
                        &full_output.textures_delta,
                        gpu,
                    );

                    {
                        let mut pass = encoder.render(
                            "egui",
                            blade_graphics::RenderTargetSet {
                                colors: &[blade_graphics::RenderTarget {
                                    view: frame.texture_view(),
                                    init_op: blade_graphics::InitOp::Clear(
                                        blade_graphics::TextureColor::TransparentBlack,
                                    ),
                                    finish_op: blade_graphics::FinishOp::Store,
                                }],
                                depth_stencil: None,
                            },
                        );
                        let sd = blade_egui::ScreenDescriptor {
                            physical_size: self.window_size,
                            scale_factor: 1.0,
                        };
                        gui_painter.paint(&mut pass, &paint_jobs, &sd, gpu);
                    }

                    encoder.present(frame);
                    let sync = gpu.submit(&mut encoder);
                    gui_painter.after_submit(&sync);
                    gpu.destroy_command_encoder(&mut encoder);

                    if let Some(ref w) = self.window {
                        w.request_redraw();
                    }
                }
                _ => {}
            }
        }
    }

    let event_loop = EventLoop::new().expect("create event loop");
    let mut harness = Harness {
        window: None,
        gpu: None,
        surface: None,
        gui_painter: None,
        egui_ctx: egui::Context::default(),
        app: QompasApp::new(),
        window_size: (1200, 700),
    };
    event_loop.run_app(&mut harness).expect("event loop");

    // Cleanup
    if let (Some(ref gpu), Some(ref mut painter)) = (&harness.gpu, &mut harness.gui_painter) {
        painter.destroy(gpu);
    }
}

/// Headless CLI demo (no GPU required).
fn cli_demo() {
    use qompas::circuit::Circuit;
    use qompas::formats::openqasm;

    println!("=== Qompas — Quantum Exploration Library (headless mode) ===\n");

    let mut bell = Circuit::new(2);
    bell.h(0).cnot(0, 1);
    let result = bell.run();
    println!("Bell state:");
    print_probs(&result.state.probabilities(), 2);

    let qasm = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n";
    let circuit = openqasm::parse(qasm).expect("valid QASM");
    let result = circuit.run();
    println!("\nBell state (from OpenQASM):");
    print_probs(&result.state.probabilities(), 2);

    let mut ghz = Circuit::new(3);
    ghz.h(0).cnot(0, 1).cnot(0, 2);
    let result = ghz.run();
    println!("\nGHZ state:");
    print_probs(&result.state.probabilities(), 3);

    println!("\nRun with a display server for the interactive visualizer.");
}

fn print_probs(probs: &[f64], n: usize) {
    for (i, p) in probs.iter().enumerate() {
        if *p > 1e-10 {
            println!("  |{:0>width$b}⟩  {:.4}", i, p, width = n);
        }
    }
}
