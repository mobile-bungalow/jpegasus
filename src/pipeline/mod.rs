use crate::types::BitDepth;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct DctPushConstants {
    pub width: u32,
    pub height: u32,
    pub pass_index: u32,
    pub block_size: u32,
    pub quantization_step: f32, // Precomputed from quality: ((101-q)/100)^3 * 2
    pub coefficient_threshold: f32,
    pub blend_original: f32,
    pub error_rate: f32,
    pub error_brightness_min: f32,
    pub error_brightness_max: f32,
    pub error_blue_yellow_min: f32,
    pub error_blue_yellow_max: f32,
    pub error_red_cyan_min: f32,
    pub error_red_cyan_max: f32,
    pub seed: u32,
    pub error_matte_mode: u32,
    pub use_error_matte: u32,
    pub use_luma_quality: u32,
    pub ae_channel_order: u32,
    pub chroma_subsampling: u32,
}

const _: () = assert!(std::mem::size_of::<DctPushConstants>() == 80);

impl DctPushConstants {
    /// Precompute quantization step from quality (1-100)
    pub fn set_quality(&mut self, quality: f32) {
        let compression = (101.0 - quality) / 100.0;
        self.quantization_step = compression * compression * compression * 2.0;
    }
}
const _: () = assert!(std::mem::align_of::<DctPushConstants>() == 4);

const WORKGROUP_SIZE: u32 = 16;
const MAX_BLOCK_SIZE: u32 = 64;

/// Get wgpu texture format for a given bit depth.
/// For input textures we use formats that match the native data.
fn input_texture_format(bit_depth: BitDepth) -> wgpu::TextureFormat {
    match bit_depth {
        BitDepth::U8 => wgpu::TextureFormat::Rgba8Unorm,
        // AE 16-bit uses 0-32768 range, use Rgba16Unorm and let shader handle it
        BitDepth::U16 => wgpu::TextureFormat::Rgba16Unorm,
        BitDepth::F32 => wgpu::TextureFormat::Rgba32Float,
        BitDepth::Invalid(_) => wgpu::TextureFormat::Rgba8Unorm,
    }
}

pub struct DctPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    dummy_texture: wgpu::Texture,

    // Cached textures - recreated only when dimensions/format change
    cached_input: Option<CachedTexture>,
    cached_output: Option<CachedTexture>,
    cached_staging: Option<CachedBuffer>,

    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

struct CachedTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

struct CachedBuffer {
    buffer: wgpu::Buffer,
    size: u64,
}

impl std::fmt::Debug for DctPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DctPipeline")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl DctPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../resources/dct.wgsl"));

        let push_constant_range = wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..std::mem::size_of::<DctPushConstants>() as u32,
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dct_layout"),
            entries: &[
                texture_entry(0),
                storage_entry(1),
                texture_entry(2),
                texture_entry(3),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dct_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: std::slice::from_ref(&push_constant_range),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dct_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("dct_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dummy_texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        Self {
            pipeline,
            bind_group_layout,
            dummy_texture,
            cached_input: None,
            cached_output: None,
            cached_staging: None,
            width: 0,
            height: 0,
            format: wgpu::TextureFormat::Rgba8Unorm,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mut params: DctPushConstants,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        input_row_bytes: usize,
        output_row_bytes: usize,
        bit_depth: BitDepth,
        error_matte: Option<&[u8]>,
        error_matte_row_bytes: Option<usize>,
        luma_quality_matte: Option<&[u8]>,
        luma_quality_row_bytes: Option<usize>,
    ) {
        params.width = width;
        params.height = height;
        // Clamp block_size to valid range (must be power of 2, max 64)
        params.block_size = params.block_size.clamp(8, MAX_BLOCK_SIZE);

        let format = input_texture_format(bit_depth);
        let bytes_per_pixel = bit_depth.bytes_per_pixel();
        let gpu_row_bytes = (width as usize) * bytes_per_pixel;

        // Ensure cached textures match current dimensions/format
        self.ensure_textures(device, width, height, format);

        // Pack and upload input
        let packed_input = pack_rows(input, width, height, input_row_bytes, bytes_per_pixel);
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.cached_input.as_ref().unwrap().texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &packed_input,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(gpu_row_bytes as u32),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // Handle error matte (not cached - can be different layer each frame)
        let error_matte_tex;
        let error_matte_view = if let Some(matte_data) = error_matte {
            let matte_row_bytes = error_matte_row_bytes.unwrap_or(input_row_bytes);
            let packed_matte =
                pack_rows(matte_data, width, height, matte_row_bytes, bytes_per_pixel);
            error_matte_tex = create_input_texture(device, width, height, format);
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &error_matte_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &packed_matte,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(gpu_row_bytes as u32),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            error_matte_tex.create_view(&wgpu::TextureViewDescriptor::default())
        } else {
            self.dummy_texture
                .create_view(&wgpu::TextureViewDescriptor::default())
        };

        // Handle luma quality matte (not cached - can be different layer each frame)
        let luma_quality_tex;
        let luma_quality_view = if let Some(matte_data) = luma_quality_matte {
            let matte_row_bytes = luma_quality_row_bytes.unwrap_or(input_row_bytes);
            let packed_matte =
                pack_rows(matte_data, width, height, matte_row_bytes, bytes_per_pixel);
            luma_quality_tex = create_input_texture(device, width, height, format);
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &luma_quality_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &packed_matte,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(gpu_row_bytes as u32),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            luma_quality_tex.create_view(&wgpu::TextureViewDescriptor::default())
        } else {
            self.dummy_texture
                .create_view(&wgpu::TextureViewDescriptor::default())
        };

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dct_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &self.cached_input.as_ref().unwrap().view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &self.cached_output.as_ref().unwrap().view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&error_matte_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&luma_quality_view),
                },
            ],
        });

        let workgroups_x = width.div_ceil(params.block_size);
        let workgroups_y = height.div_ceil(params.block_size);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dct_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dct_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.set_push_constants(0, bytemuck::bytes_of(&params));
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Output is Rgba16Float (8 bytes per pixel)
        let output_bytes_per_pixel = 8u32;
        let gpu_output_row_bytes = width * output_bytes_per_pixel;
        let padded_row_bytes = (gpu_output_row_bytes + 255) & !255;

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.cached_output.as_ref().unwrap().texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.cached_staging.as_ref().unwrap().buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));

        let staging_buffer = &self.cached_staging.as_ref().unwrap().buffer;
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let mapped = buffer_slice.get_mapped_range();

        // Convert from Rgba16Float directly to target format
        convert_f16_to_output(
            &mapped,
            output,
            width,
            height,
            padded_row_bytes as usize,
            output_row_bytes,
            bit_depth,
        );

        drop(mapped);
        staging_buffer.unmap();
    }

    /// Ensure cached textures and staging buffer exist with correct dimensions/format
    fn ensure_textures(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) {
        // Check if input texture needs recreation
        let needs_input = self.cached_input.as_ref().map_or(true, |c| {
            c.width != width || c.height != height || c.format != format
        });
        if needs_input {
            let texture = create_input_texture(device, width, height, format);
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.cached_input = Some(CachedTexture {
                texture,
                view,
                width,
                height,
                format,
            });
        }

        // Check if output texture needs recreation
        let needs_output = self
            .cached_output
            .as_ref()
            .map_or(true, |c| c.width != width || c.height != height);
        if needs_output {
            let texture = create_output_texture(device, width, height);
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.cached_output = Some(CachedTexture {
                texture,
                view,
                width,
                height,
                format: wgpu::TextureFormat::Rgba16Float,
            });
        }

        // Check if staging buffer needs recreation
        let output_bytes_per_pixel = 8u32;
        let gpu_output_row_bytes = width * output_bytes_per_pixel;
        let padded_row_bytes = (gpu_output_row_bytes + 255) & !255;
        let buffer_size = padded_row_bytes as u64 * height as u64;

        let needs_staging = self
            .cached_staging
            .as_ref()
            .map_or(true, |c| c.size < buffer_size);
        if needs_staging {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging_buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.cached_staging = Some(CachedBuffer {
                buffer,
                size: buffer_size,
            });
        }

        self.width = width;
        self.height = height;
        self.format = format;
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba16Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn create_output_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn create_input_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("input_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// Pack rows from AE buffer (with potential padding) into tight GPU buffer
fn pack_rows(
    input: &[u8],
    width: u32,
    height: u32,
    src_row_bytes: usize,
    bytes_per_pixel: usize,
) -> Vec<u8> {
    let width = width as usize;
    let height = height as usize;
    let dst_row_bytes = width * bytes_per_pixel;

    // Fast path: if strides match, just return a copy
    if src_row_bytes == dst_row_bytes {
        return input[..height * dst_row_bytes].to_vec();
    }

    let mut out = vec![0u8; height * dst_row_bytes];
    for y in 0..height {
        let src_start = y * src_row_bytes;
        let dst_start = y * dst_row_bytes;
        out[dst_start..dst_start + dst_row_bytes]
            .copy_from_slice(&input[src_start..src_start + dst_row_bytes]);
    }
    out
}

/// Convert Rgba16Float GPU output directly to the target bit depth format.
/// This avoids the intermediate 8-bit buffer entirely.
fn convert_f16_to_output(
    gpu_data: &[u8],
    output: &mut [u8],
    width: u32,
    height: u32,
    gpu_row_bytes: usize,
    dst_row_bytes: usize,
    bit_depth: BitDepth,
) {
    let width = width as usize;
    let height = height as usize;

    for y in 0..height {
        let src_row_start = y * gpu_row_bytes;
        let dst_row_start = y * dst_row_bytes;

        match bit_depth {
            BitDepth::U8 => {
                for x in 0..width {
                    let src_px = src_row_start + x * 8;
                    let dst_px = dst_row_start + x * 4;
                    for c in 0..4 {
                        let f16_bits = u16::from_le_bytes([
                            gpu_data[src_px + c * 2],
                            gpu_data[src_px + c * 2 + 1],
                        ]);
                        let f32_val = half::f16::from_bits(f16_bits).to_f32();
                        output[dst_px + c] = (f32_val.clamp(0.0, 1.0) * 255.0) as u8;
                    }
                }
            }
            BitDepth::U16 => {
                // AE 16-bit uses 0-32768 range (15-bit + 1)
                for x in 0..width {
                    let src_px = src_row_start + x * 8;
                    let dst_px = dst_row_start + x * 8;
                    for c in 0..4 {
                        let f16_bits = u16::from_le_bytes([
                            gpu_data[src_px + c * 2],
                            gpu_data[src_px + c * 2 + 1],
                        ]);
                        let f32_val = half::f16::from_bits(f16_bits).to_f32();
                        // Scale to AE's 0-32768 range
                        let u16_val = (f32_val.clamp(0.0, 1.0) * 32768.0) as u16;
                        let bytes = u16_val.to_ne_bytes();
                        output[dst_px + c * 2] = bytes[0];
                        output[dst_px + c * 2 + 1] = bytes[1];
                    }
                }
            }
            BitDepth::F32 => {
                for x in 0..width {
                    let src_px = src_row_start + x * 8;
                    let dst_px = dst_row_start + x * 16;
                    for c in 0..4 {
                        let f16_bits = u16::from_le_bytes([
                            gpu_data[src_px + c * 2],
                            gpu_data[src_px + c * 2 + 1],
                        ]);
                        let f32_val = half::f16::from_bits(f16_bits).to_f32();
                        let bytes = f32_val.to_ne_bytes();
                        output[dst_px + c * 4] = bytes[0];
                        output[dst_px + c * 4 + 1] = bytes[1];
                        output[dst_px + c * 4 + 2] = bytes[2];
                        output[dst_px + c * 4 + 3] = bytes[3];
                    }
                }
            }
            BitDepth::Invalid(_) => {
                // Fallback: treat as U8
                for x in 0..width {
                    let src_px = src_row_start + x * 8;
                    let dst_px = dst_row_start + x * 4;
                    for c in 0..4 {
                        let f16_bits = u16::from_le_bytes([
                            gpu_data[src_px + c * 2],
                            gpu_data[src_px + c * 2 + 1],
                        ]);
                        let f32_val = half::f16::from_bits(f16_bits).to_f32();
                        output[dst_px + c] = (f32_val.clamp(0.0, 1.0) * 255.0) as u8;
                    }
                }
            }
        }
    }
}
