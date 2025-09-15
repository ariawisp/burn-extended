//! WGPU + winit example for Matrix-Game-2 control + rendering (Metal/DirectX/Vulkan).
//! Minimal pipeline: clear + fullscreen triangle with a color driven by the last control token.

use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowBuilder;

fn key_to_token(code: KeyCode) -> Option<u32> {
    match code {
        KeyCode::ArrowUp | KeyCode::KeyW => Some(1),
        KeyCode::ArrowDown | KeyCode::KeyS => Some(2),
        KeyCode::ArrowLeft | KeyCode::KeyA => Some(3),
        KeyCode::ArrowRight | KeyCode::KeyD => Some(4),
        _ => None,
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PushColor {
    rgba: [f32; 4],
}

fn token_to_color(tok: u32) -> [f32; 4] {
    match tok {
        1 => [0.2, 0.7, 1.0, 1.0], // up
        2 => [1.0, 0.4, 0.2, 1.0], // down
        3 => [0.6, 0.2, 1.0, 1.0], // left
        4 => [0.2, 1.0, 0.4, 1.0], // right
        _ => [0.1, 0.1, 0.1, 1.0],
    }
}

static VS: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    let xy = pos[vertex_index];
    return vec4<f32>(xy, 0.0, 1.0);
}
"#;

static FS: &str = r#"
struct PushColor { rgba: vec4<f32> };
@group(0) @binding(0) var<uniform> u_color: PushColor;

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return u_color.rgba;
}
"#;

fn main() -> anyhow::Result<()> {
    // Window
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new().with_title("Matrix-Game-2 (wgpu)").build(&event_loop)?;

    // Instance + surface
    let instance = wgpu::Instance::default();
    let surface = unsafe { instance.create_surface(&window)? };
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| anyhow::anyhow!("No suitable GPU adapter found"))?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("device"),
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))?;

    // Configure surface
    let size = window.inner_size();
    let surface_caps = surface.get_capabilities(&adapter);
    let format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: wgpu::PresentMode::Fifo,
        desired_maximum_frame_latency: 2,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
    };
    surface.configure(&device, &config);

    // Pipeline
    let vs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vs"),
        source: wgpu::ShaderSource::Wgsl(VS.into()),
    });
    let fs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fs"),
        source: wgpu::ShaderSource::Wgsl(FS.into()),
    });

    // Uniform for color
    let color = PushColor { rgba: [0.1, 0.1, 0.1, 1.0] };
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("color_ubo"),
        contents: bytemuck::bytes_of(&color),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bind_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(std::num::NonZeroU64::new(std::mem::size_of::<PushColor>() as u64).unwrap()),
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bind_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState { module: &vs_mod, entry_point: "vs_main", buffers: &[] },
        fragment: Some(wgpu::FragmentState { module: &fs_mod, entry_point: "fs_main", targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })] }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    // Event state
    let mut tokens: Vec<u32> = Vec::new();
    let tick = Duration::from_millis(33);
    let mut last_tick = Instant::now();

    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
                config.width = new_size.width.max(1);
                config.height = new_size.height.max(1);
                surface.configure(&device, &config);
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(code), state, .. }, .. }, .. } => {
                if let ElementState::Pressed = state {
                    if let Some(tok) = key_to_token(code) {
                        tokens.push(tok);
                    }
                }
            }
            Event::AboutToWait => {
                if last_tick.elapsed() >= tick {
                    last_tick = Instant::now();
                    window.request_redraw();
                }
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let frame = match surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        surface.configure(&device, &config);
                        return;
                    }
                    Err(e) => {
                        eprintln!("surface error: {e}");
                        elwt.exit();
                        return;
                    }
                };
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

                // Update color from last token
                let color = tokens.last().copied().map(token_to_color).unwrap_or([0.1, 0.1, 0.1, 1.0]);
                queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&PushColor { rgba: color }));

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("encoder") });
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("rpass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: true },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    rpass.set_pipeline(&pipeline);
                    rpass.set_bind_group(0, &bind_group, &[]);
                    rpass.draw(0..3, 0..1); // fullscreen triangle
                }
                queue.submit([encoder.finish()]);
                frame.present();
            }
            _ => {}
        }
    })
}

