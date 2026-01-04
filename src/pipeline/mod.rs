use crate::types::BitDepth;
use bytemuck::{Pod, Zeroable};

pub struct Layer<'a> {
    pub buffer: &'a [u8],
    pub row_bytes: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct DctPushConstants {
    pub width: u32,
    pub height: u32,
    pub pass_index: u32,
    pub block_size: u32,
    pub quantization_step: f32,
    pub coefficient_min: f32,
    pub coefficient_max: f32,
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

const _: () = assert!(std::mem::size_of::<DctPushConstants>() == 84);
const _: () = assert!(std::mem::align_of::<DctPushConstants>() == 4);

impl DctPushConstants {
    pub fn new() -> Self {
        Self {
            coefficient_max: 1.0,
            ..Self::zeroed()
        }
    }

    fn zeroed() -> Self {
        bytemuck::Zeroable::zeroed()
    }

    pub fn set_quality(&mut self, quality: f32) {
        let compression = (101.0 - quality) / 100.0;
        self.quantization_step = compression * compression * compression * 2.0;
    }
}

fn input_texture_format(bit_depth: BitDepth) -> wgpu::TextureFormat {
    match bit_depth {
        BitDepth::U8 => wgpu::TextureFormat::Rgba8Unorm,
        BitDepth::U16 => wgpu::TextureFormat::Rgba16Unorm,
        BitDepth::F32 => wgpu::TextureFormat::Rgba32Float,
        BitDepth::Invalid(_) => wgpu::TextureFormat::Rgba8Unorm,
    }
}

pub struct DctPipeline {
    // Multi-pass pipelines
    pass_rgb_to_ycbcr: wgpu::ComputePipeline,
    pass_dct_rows: wgpu::ComputePipeline,
    pass_dct_cols: wgpu::ComputePipeline,
    pass_quantize: wgpu::ComputePipeline,
    pass_idct_cols: wgpu::ComputePipeline,
    pass_idct_rows: wgpu::ComputePipeline,
    pass_finalize: wgpu::ComputePipeline,
    pass_finalize_8bit: wgpu::ComputePipeline,

    bind_group_layout_2tex: wgpu::BindGroupLayout,
    bind_group_layout_5tex: wgpu::BindGroupLayout,
    bind_group_layout_6tex: wgpu::BindGroupLayout,

    dummy_texture: wgpu::Texture,
    dummy_view: wgpu::TextureView,

    // Cached resources
    cached_input: Option<CachedTexture>,
    cached_output: Option<CachedTexture>,
    cached_output_8bit: Option<CachedTexture>,
    cached_scratch: [Option<CachedTexture>; 2],
    cached_staging: Option<CachedBuffer>,
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
        f.debug_struct("DctPipeline").finish_non_exhaustive()
    }
}

impl DctPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../resources/dct.wgsl"));

        let push_constant_range = wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..std::mem::size_of::<DctPushConstants>() as u32,
        };

        // Layout for simple passes: input texture + output storage
        let bind_group_layout_2tex =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dct_layout_2tex"),
                entries: &[texture_entry(0), storage_entry(1)],
            });

        // Layout for finalize pass: input + output + original + error_matte + quality_matte
        let bind_group_layout_5tex =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dct_layout_5tex"),
                entries: &[
                    texture_entry(0),
                    storage_entry(1),
                    texture_entry(2),
                    texture_entry(3),
                    texture_entry(4),
                ],
            });

        // Layout for 8-bit finalize: same as 5tex but with 8-bit output at binding 5
        let bind_group_layout_6tex =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dct_layout_6tex"),
                entries: &[
                    texture_entry(0),
                    storage_entry(1),
                    texture_entry(2),
                    texture_entry(3),
                    texture_entry(4),
                    storage_entry_format(5, wgpu::TextureFormat::Rgba8Unorm),
                ],
            });

        let layout_2tex = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dct_layout_2tex"),
            bind_group_layouts: &[&bind_group_layout_2tex],
            push_constant_ranges: &[push_constant_range.clone()],
        });

        let layout_5tex = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dct_layout_5tex"),
            bind_group_layouts: &[&bind_group_layout_5tex],
            push_constant_ranges: &[push_constant_range.clone()],
        });

        let layout_6tex = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dct_layout_6tex"),
            bind_group_layouts: &[&bind_group_layout_6tex],
            push_constant_ranges: &[push_constant_range],
        });

        let make_pipeline = |layout: &wgpu::PipelineLayout, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let pass_rgb_to_ycbcr = make_pipeline(&layout_2tex, "pass_rgb_to_ycbcr");
        let pass_dct_rows = make_pipeline(&layout_2tex, "pass_dct_rows");
        let pass_dct_cols = make_pipeline(&layout_2tex, "pass_dct_cols");
        let pass_quantize = make_pipeline(&layout_5tex, "pass_quantize");
        let pass_idct_cols = make_pipeline(&layout_2tex, "pass_idct_cols");
        let pass_idct_rows = make_pipeline(&layout_2tex, "pass_idct_rows");
        let pass_finalize = make_pipeline(&layout_5tex, "pass_finalize");
        let pass_finalize_8bit = make_pipeline(&layout_6tex, "pass_finalize_8bit");

        let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            pass_rgb_to_ycbcr,
            pass_dct_rows,
            pass_dct_cols,
            pass_quantize,
            pass_idct_cols,
            pass_idct_rows,
            pass_finalize,
            pass_finalize_8bit,
            bind_group_layout_2tex,
            bind_group_layout_5tex,
            bind_group_layout_6tex,
            dummy_texture,
            dummy_view,
            cached_input: None,
            cached_output: None,
            cached_output_8bit: None,
            cached_scratch: [None, None],
            cached_staging: None,
        }
    }

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
        error_matte: Option<Layer>,
        luma_quality_matte: Option<Layer>,
    ) {
        params.width = width;
        params.height = height;
        params.block_size = params.block_size.clamp(2, 64);

        let format = input_texture_format(bit_depth);
        let bytes_per_pixel = bit_depth.bytes_per_pixel();
        let gpu_row_bytes = (width as usize) * bytes_per_pixel;

        self.ensure_textures(device, width, height, format);

        // Upload input
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

        // Upload mattes
        let error_matte_view = self.upload_matte(
            device,
            queue,
            error_matte.as_ref(),
            width,
            height,
            format,
            bytes_per_pixel,
            gpu_row_bytes,
        );
        let quality_matte_view = self.upload_matte(
            device,
            queue,
            luma_quality_matte.as_ref(),
            width,
            height,
            format,
            bytes_per_pixel,
            gpu_row_bytes,
        );

        let workgroups_x = width.div_ceil(8);
        let workgroups_y = height.div_ceil(8);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dct") });

        let input_view = &self.cached_input.as_ref().unwrap().view;
        let output_view = &self.cached_output.as_ref().unwrap().view;
        let scratch0_view = &self.cached_scratch[0].as_ref().unwrap().view;
        let scratch1_view = &self.cached_scratch[1].as_ref().unwrap().view;

        // Pass 0: RGB → YCbCr (input → scratch0)
        self.dispatch_2tex(
            &mut encoder,
            &self.pass_rgb_to_ycbcr,
            input_view,
            scratch0_view,
            &params,
            workgroups_x,
            workgroups_y,
            device,
        );

        // Pass 1: DCT rows (scratch0 → scratch1)
        self.dispatch_2tex(
            &mut encoder,
            &self.pass_dct_rows,
            scratch0_view,
            scratch1_view,
            &params,
            workgroups_x,
            workgroups_y,
            device,
        );

        // Pass 2: DCT cols (scratch1 → scratch0)
        self.dispatch_2tex(
            &mut encoder,
            &self.pass_dct_cols,
            scratch1_view,
            scratch0_view,
            &params,
            workgroups_x,
            workgroups_y,
            device,
        );

        // Pass 3: Quantize (scratch0 → scratch1, needs quality matte)
        self.dispatch_quantize(
            &mut encoder,
            scratch0_view,
            scratch1_view,
            &quality_matte_view,
            &params,
            workgroups_x,
            workgroups_y,
            device,
        );

        // Pass 4: IDCT cols (scratch1 → scratch0)
        self.dispatch_2tex(
            &mut encoder,
            &self.pass_idct_cols,
            scratch1_view,
            scratch0_view,
            &params,
            workgroups_x,
            workgroups_y,
            device,
        );

        // Pass 5: IDCT rows (scratch0 → scratch1)
        self.dispatch_2tex(
            &mut encoder,
            &self.pass_idct_rows,
            scratch0_view,
            scratch1_view,
            &params,
            workgroups_x,
            workgroups_y,
            device,
        );

        // Pass 6: Finalize (scratch1 → output, needs original + error matte)
        let use_8bit = matches!(bit_depth, BitDepth::U8);
        let output_8bit_view = &self.cached_output_8bit.as_ref().unwrap().view;

        if use_8bit {
            self.dispatch_finalize_8bit(
                &mut encoder,
                scratch1_view,
                output_view,
                output_8bit_view,
                input_view,
                &error_matte_view,
                &quality_matte_view,
                &params,
                workgroups_x,
                workgroups_y,
                device,
            );
        } else {
            self.dispatch_finalize(
                &mut encoder,
                scratch1_view,
                output_view,
                input_view,
                &error_matte_view,
                &quality_matte_view,
                &params,
                workgroups_x,
                workgroups_y,
                device,
            );
        }

        // Copy output to staging buffer
        let output_bytes_per_pixel = if use_8bit { 4u32 } else { 8u32 };
        let gpu_output_row_bytes = width * output_bytes_per_pixel;
        let padded_row_bytes = (gpu_output_row_bytes + 255) & !255;

        let output_texture = if use_8bit {
            &self.cached_output_8bit.as_ref().unwrap().texture
        } else {
            &self.cached_output.as_ref().unwrap().texture
        };

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: output_texture,
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

        // Read back
        let staging_buffer = &self.cached_staging.as_ref().unwrap().buffer;
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let mapped = buffer_slice.get_mapped_range();
        if use_8bit {
            copy_8bit_to_output(
                &mapped,
                output,
                width,
                height,
                padded_row_bytes as usize,
                output_row_bytes,
            );
        } else {
            convert_f16_to_output(
                &mapped,
                output,
                width,
                height,
                padded_row_bytes as usize,
                output_row_bytes,
                bit_depth,
            );
        }
        drop(mapped);
        staging_buffer.unmap();
    }

    fn dispatch_2tex(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
        params: &DctPushConstants,
        wg_x: u32,
        wg_y: u32,
        device: &wgpu::Device,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout_2tex,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(params));
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    fn dispatch_quantize(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
        quality_matte: &wgpu::TextureView,
        params: &DctPushConstants,
        wg_x: u32,
        wg_y: u32,
        device: &wgpu::Device,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout_5tex,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(quality_matte),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pass_quantize);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(params));
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    fn dispatch_finalize(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
        original: &wgpu::TextureView,
        error_matte: &wgpu::TextureView,
        quality_matte: &wgpu::TextureView,
        params: &DctPushConstants,
        wg_x: u32,
        wg_y: u32,
        device: &wgpu::Device,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout_5tex,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(original),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(error_matte),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(quality_matte),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pass_finalize);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(params));
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    fn dispatch_finalize_8bit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output_16bit: &wgpu::TextureView,
        output_8bit: &wgpu::TextureView,
        original: &wgpu::TextureView,
        error_matte: &wgpu::TextureView,
        quality_matte: &wgpu::TextureView,
        params: &DctPushConstants,
        wg_x: u32,
        wg_y: u32,
        device: &wgpu::Device,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout_6tex,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_16bit),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(original),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(error_matte),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(quality_matte),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(output_8bit),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pass_finalize_8bit);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(params));
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    fn upload_matte(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layer: Option<&Layer>,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        bytes_per_pixel: usize,
        gpu_row_bytes: usize,
    ) -> wgpu::TextureView {
        let Some(layer) = layer else {
            return self
                .dummy_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
        };

        let packed = pack_rows(
            layer.buffer,
            width,
            height,
            layer.row_bytes,
            bytes_per_pixel,
        );
        let texture = create_input_texture(device, width, height, format);
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &packed,
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
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn ensure_textures(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) {
        // Input texture
        if self
            .cached_input
            .as_ref()
            .is_none_or(|c| c.width != width || c.height != height || c.format != format)
        {
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

        // Output texture (16-bit float)
        if self
            .cached_output
            .as_ref()
            .is_none_or(|c| c.width != width || c.height != height)
        {
            let texture =
                create_storage_texture(device, width, height, wgpu::TextureFormat::Rgba16Float);
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.cached_output = Some(CachedTexture {
                texture,
                view,
                width,
                height,
                format: wgpu::TextureFormat::Rgba16Float,
            });
        }

        // Output texture (8-bit)
        if self
            .cached_output_8bit
            .as_ref()
            .is_none_or(|c| c.width != width || c.height != height)
        {
            let texture =
                create_storage_texture(device, width, height, wgpu::TextureFormat::Rgba8Unorm);
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.cached_output_8bit = Some(CachedTexture {
                texture,
                view,
                width,
                height,
                format: wgpu::TextureFormat::Rgba8Unorm,
            });
        }

        // Scratch textures (need both read and write)
        for i in 0..2 {
            if self.cached_scratch[i]
                .as_ref()
                .is_none_or(|c| c.width != width || c.height != height)
            {
                let texture = create_scratch_texture(device, width, height);
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                self.cached_scratch[i] = Some(CachedTexture {
                    texture,
                    view,
                    width,
                    height,
                    format: wgpu::TextureFormat::Rgba16Float,
                });
            }
        }

        // Staging buffer
        let output_bytes_per_pixel = 8u32;
        let gpu_output_row_bytes = width * output_bytes_per_pixel;
        let padded_row_bytes = (gpu_output_row_bytes + 255) & !255;
        let buffer_size = padded_row_bytes as u64 * height as u64;

        if self
            .cached_staging
            .as_ref()
            .is_none_or(|c| c.size < buffer_size)
        {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.cached_staging = Some(CachedBuffer {
                buffer,
                size: buffer_size,
            });
        }
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
    storage_entry_format(binding, wgpu::TextureFormat::Rgba16Float)
}

fn storage_entry_format(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn create_input_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("input"),
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

fn create_storage_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn create_scratch_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scratch"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    })
}

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

fn copy_8bit_to_output(
    gpu_data: &[u8],
    output: &mut [u8],
    width: u32,
    height: u32,
    gpu_row_bytes: usize,
    dst_row_bytes: usize,
) {
    let width = width as usize;
    let height = height as usize;
    let pixel_bytes = 4usize;

    if gpu_row_bytes == dst_row_bytes {
        output[..height * dst_row_bytes].copy_from_slice(&gpu_data[..height * gpu_row_bytes]);
    } else {
        for y in 0..height {
            let src_start = y * gpu_row_bytes;
            let dst_start = y * dst_row_bytes;
            let row_len = width * pixel_bytes;
            output[dst_start..dst_start + row_len]
                .copy_from_slice(&gpu_data[src_start..src_start + row_len]);
        }
    }
}

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
        let src_row = y * gpu_row_bytes;
        let dst_row = y * dst_row_bytes;

        for x in 0..width {
            let src_px = src_row + x * 8;

            match bit_depth {
                BitDepth::U8 => {
                    let dst_px = dst_row + x * 4;
                    for c in 0..4 {
                        let f16_bits = u16::from_le_bytes([
                            gpu_data[src_px + c * 2],
                            gpu_data[src_px + c * 2 + 1],
                        ]);
                        output[dst_px + c] =
                            (half::f16::from_bits(f16_bits).to_f32().clamp(0.0, 1.0) * 255.0) as u8;
                    }
                }
                BitDepth::U16 => {
                    // AE 16-bit uses 0-32768 range where 32768 = 1.0
                    // But Rgba16Unorm input normalizes 65535 = 1.0, so input is halved
                    // Compensate by outputting to full 32768 range
                    let dst_px = dst_row + x * 8;
                    for c in 0..4 {
                        let f16_bits = u16::from_le_bytes([
                            gpu_data[src_px + c * 2],
                            gpu_data[src_px + c * 2 + 1],
                        ]);
                        // Multiply by 2 to compensate for input being halved, then scale to 32768
                        let u16_val = (half::f16::from_bits(f16_bits).to_f32().clamp(0.0, 1.0)
                            * 2.0
                            * 32768.0)
                            .min(32768.0) as u16;
                        let bytes = u16_val.to_ne_bytes();
                        output[dst_px + c * 2] = bytes[0];
                        output[dst_px + c * 2 + 1] = bytes[1];
                    }
                }
                BitDepth::F32 => {
                    let dst_px = dst_row + x * 16;
                    for c in 0..4 {
                        let f16_bits = u16::from_le_bytes([
                            gpu_data[src_px + c * 2],
                            gpu_data[src_px + c * 2 + 1],
                        ]);
                        let bytes = half::f16::from_bits(f16_bits).to_f32().to_ne_bytes();
                        output[dst_px + c * 4..dst_px + c * 4 + 4].copy_from_slice(&bytes);
                    }
                }
                BitDepth::Invalid(_) => {
                    let dst_px = dst_row + x * 4;
                    for c in 0..4 {
                        let f16_bits = u16::from_le_bytes([
                            gpu_data[src_px + c * 2],
                            gpu_data[src_px + c * 2 + 1],
                        ]);
                        output[dst_px + c] =
                            (half::f16::from_bits(f16_bits).to_f32().clamp(0.0, 1.0) * 255.0) as u8;
                    }
                }
            }
        }
    }
}
