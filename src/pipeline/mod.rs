mod convert;

use crate::types::BitDepth;
use bytemuck::{Pod, Zeroable};
use wgpu::*;

pub struct Layer<'a> {
    pub buffer: &'a [u8],
    pub row_bytes: usize,
    pub width: u32,
    pub height: u32,
    pub bit_depth: BitDepth,
}

pub struct LayerMut<'a> {
    pub buffer: &'a mut [u8],
    pub row_bytes: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct DctPushConstants {
    pub width: u32,
    pub height: u32,
    pub block_size: u32,
    pub quantization_step: f32,
    pub coefficient_min: f32,
    pub coefficient_max: f32,
    pub blend_original: f32,
    pub use_luma_quality: u32,
    pub ae_channel_order: u32,
    pub chroma_subsampling: u32, // 0=4:4:4, 1=4:2:2, 2=4:2:0, 3=4:1:1
}

impl DctPushConstants {
    pub fn new() -> Self {
        Self {
            coefficient_max: 1.0,
            ..Zeroable::zeroed()
        }
    }

    /// Maps quality 1-100 to quantization step.
    /// Uses cubic curve for perceptual scaling.
    pub fn set_quality(&mut self, quality: f32) {
        let c = (101.0 - quality) / 100.0;
        self.quantization_step = c * c * c * 2.0;
    }
}

fn create_shader(device: &Device) -> ShaderModule {
    device.create_shader_module(include_wgsl!("../shaders/dct_pipeline.wgsl"))
}

pub struct DctPipeline {
    pipelines: [ComputePipeline; 5], // 5 passes, single 8-bit output
    layout: BindGroupLayout,
    cache: Option<Cache>,
}

struct Cache {
    w: u32,
    h: u32,
    input: (Texture, TextureView),
    output: (Texture, TextureView),
    dct: [(Texture, TextureView); 2],
    staging: Buffer,
    staging_mapped: bool,
    upload_buf: Vec<u8>,
}

impl std::fmt::Debug for DctPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DctPipeline").finish_non_exhaustive()
    }
}

impl DctPipeline {
    pub fn new(device: &Device) -> Self {
        let shader = create_shader(device);

        // Binding layout:
        // 0: input texture (read)
        // 1: dct read texture
        // 2: dct write texture
        // 3: quality matte texture
        // 4: 8-bit output
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                tex_entry(0),
                tex_entry(1),
                storage_entry(2, TextureFormat::Rgba16Float),
                tex_entry(3),
                storage_entry(4, TextureFormat::Rgba8Unorm),
            ],
        });

        let pipe_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            immediate_size: std::mem::size_of::<DctPushConstants>() as u32,
        });

        // 5 passes: DCT rows (with RGB->YCbCr), DCT cols, quantize, IDCT cols, IDCT rows (with YCbCr->RGB + finalize)
        let entries = [
            "pass_dct_rows",
            "pass_dct_cols",
            "pass_quantize",
            "pass_idct_cols",
            "pass_idct_rows_final",
        ];
        let pipelines = entries.map(|e| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(e),
                layout: Some(&pipe_layout),
                module: &shader,
                entry_point: Some(e),
                compilation_options: Default::default(),
                cache: None,
            })
        });

        Self {
            pipelines,
            layout,
            cache: None,
        }
    }

    pub fn render(
        &mut self,
        device: &Device,
        queue: &Queue,
        mut params: DctPushConstants,
        input: Layer,
        output: LayerMut,
        quality_matte: Option<Layer>,
    ) {
        let width = input.width;
        let height = input.height;
        let bit_depth = input.bit_depth;

        params.width = width;
        params.height = height;
        params.block_size = params.block_size.clamp(2, 64);

        self.ensure_cache(device, width, height);
        let cache = self.cache.as_mut().unwrap();

        convert::pack_rgba8(
            &mut cache.upload_buf,
            input.buffer,
            bit_depth,
            width,
            height,
            input.row_bytes,
        );
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &cache.input.0,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &cache.upload_buf,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let quality_view = if let Some(m) = quality_matte {
            convert::pack_rgba8(
                &mut cache.upload_buf,
                m.buffer,
                m.bit_depth,
                width,
                height,
                m.row_bytes,
            );
            let tex = make_tex(device, width, height, TextureFormat::Rgba8Unorm);
            queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                &cache.upload_buf,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            tex.create_view(&TextureViewDescriptor::default())
        } else {
            make_dummy_texture(device)
        };

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        let workgroups = (width.div_ceil(8), height.div_ceil(8));

        let bind = |dct_read: &TextureView, dct_write: &TextureView| {
            device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&cache.input.1),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(dct_read),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(dct_write),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::TextureView(&quality_view),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: BindingResource::TextureView(&cache.output.1),
                    },
                ],
            })
        };

        // Pass 0: input -> dct[0], Pass 1: dct[0] -> dct[1], Pass 2: dct[1] -> dct[0], etc.
        let bg_to_0 = bind(&cache.input.1, &cache.dct[0].1);
        let bg_0_to_1 = bind(&cache.dct[0].1, &cache.dct[1].1);
        let bg_1_to_0 = bind(&cache.dct[1].1, &cache.dct[0].1);
        let bind_groups = [&bg_to_0, &bg_0_to_1, &bg_1_to_0, &bg_0_to_1, &bg_1_to_0];

        for (pipeline, bind_group) in self.pipelines.iter().zip(bind_groups) {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_immediates(0, bytemuck::bytes_of(&params));
            pass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
        }

        let row_stride = (width * 4 + 255) & !255;
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &cache.output.0,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &cache.staging,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_stride),
                    rows_per_image: Some(height),
                },
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // Must unmap before submit - buffer can't be mapped during GPU work
        if cache.staging_mapped {
            cache.staging.unmap();
            cache.staging_mapped = false;
        }

        let submission = queue.submit(Some(encoder.finish()));

        let slice = cache.staging.slice(..);
        slice.map_async(MapMode::Read, |r| r.unwrap());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .ok();
        cache.staging_mapped = true;

        let gpu_output = slice.get_mapped_range();
        match bit_depth {
            BitDepth::U8 | BitDepth::Invalid(_) => convert::copy_rows(
                &gpu_output,
                output.buffer,
                width,
                height,
                row_stride as usize,
                output.row_bytes,
            ),
            BitDepth::U16 => convert::from_8bit_to_16(
                &gpu_output,
                output.buffer,
                width,
                height,
                row_stride as usize,
                output.row_bytes,
            ),
            BitDepth::F32 => convert::from_8bit_to_f32(
                &gpu_output,
                output.buffer,
                width,
                height,
                row_stride as usize,
                output.row_bytes,
            ),
        }
        drop(gpu_output);
        cache.staging.unmap();
        cache.staging_mapped = false;
    }

    fn ensure_cache(&mut self, device: &Device, w: u32, h: u32) {
        if self.cache.as_ref().is_some_and(|c| c.w == w && c.h == h) {
            return;
        }

        let input = make_tex_view(
            device,
            w,
            h,
            TextureFormat::Rgba8Unorm,
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        );
        let output = make_tex_view(
            device,
            w,
            h,
            TextureFormat::Rgba8Unorm,
            TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        );
        let dct = [
            make_tex_view(
                device,
                w,
                h,
                TextureFormat::Rgba16Float,
                TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            ),
            make_tex_view(
                device,
                w,
                h,
                TextureFormat::Rgba16Float,
                TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            ),
        ];
        let staging_size = ((w * 4 + 255) & !255) as u64 * h as u64;
        let staging = device.create_buffer(&BufferDescriptor {
            label: None,
            size: staging_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let upload_buf = vec![0u8; (w * h * 4) as usize];

        self.cache = Some(Cache {
            w,
            h,
            input,
            output,
            dct,
            staging,
            staging_mapped: false,
            upload_buf,
        });
    }
}

fn tex_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        count: None,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: false },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
    }
}

fn storage_entry(binding: u32, format: TextureFormat) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        count: None,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format,
            view_dimension: TextureViewDimension::D2,
        },
    }
}

fn make_tex(device: &Device, w: u32, h: u32, format: TextureFormat) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn make_tex_view(
    device: &Device,
    w: u32,
    h: u32,
    format: TextureFormat,
    usage: TextureUsages,
) -> (Texture, TextureView) {
    let t = device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let v = t.create_view(&TextureViewDescriptor::default());
    (t, v)
}

fn make_dummy_texture(device: &Device) -> TextureView {
    make_tex(device, 1, 1, TextureFormat::Rgba8Unorm).create_view(&TextureViewDescriptor::default())
}
