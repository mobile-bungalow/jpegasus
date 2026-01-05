mod convert;

use crate::types::BitDepth;
use bytemuck::{Pod, Zeroable};
use wgpu::{util::DeviceExt, wgt::TextureDataOrder, *};

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
    pub use_ycbcr: u32, // 0 = RGB, 1 = YCbCr
}

impl DctPushConstants {
    pub fn new() -> Self {
        Self {
            coefficient_max: 1.0,
            use_ycbcr: 1,
            ..Zeroable::zeroed()
        }
    }

    pub fn set_quality(&mut self, quality: f32) {
        let c = (101.0 - quality) / 100.0;
        self.quantization_step = c * c * c * 2.0;
    }
}

fn create_shader(device: &Device) -> ShaderModule {
    device.create_shader_module(include_wgsl!("../shaders/dct_pipeline.wgsl"))
}

pub struct DctPipeline {
    pipelines: [ComputePipeline; 5],
    layout: BindGroupLayout,
    cache: Option<Cache>,
}

struct Cache {
    w: u32,
    h: u32,
    input: (Texture, TextureView),
    output: (Texture, TextureView),
    dct: [(Texture, TextureView); 2],
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

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    count: None,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    count: None,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    count: None,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    count: None,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    count: None,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                },
            ],
        });

        let pipe_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            immediate_size: std::mem::size_of::<DctPushConstants>() as u32,
        });

        let entries = [
            "pass_dct_rows",
            "pass_dct_cols",
            "pass_quantize",
            "pass_idct_cols",
            "pass_idct_rows_final",
        ];
        let pipelines = entries.map(|entry_point| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipe_layout),
                module: &shader,
                entry_point: Some(entry_point),
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

        convert::load_rbga8_tex(
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

        let quality_view = if let Some(matte) = quality_matte {
            convert::load_rbga8_tex(
                &mut cache.upload_buf,
                matte.buffer,
                matte.bit_depth,
                matte.width,
                matte.height,
                matte.row_bytes,
            );

            let tex = device.create_texture_with_data(
                queue,
                &TextureDescriptor {
                    label: Some("Luma Matte"),
                    size: Extent3d {
                        width: matte.width,
                        height: matte.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                TextureDataOrder::LayerMajor,
                &cache.upload_buf,
            );
            tex.create_view(&TextureViewDescriptor::default())
        } else {
            // Binding placeholder
            let tex = device.create_texture(&TextureDescriptor {
                label: Some("Placeholder"),
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            tex.create_view(&TextureViewDescriptor::default())
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

        let bg_to_0 = bind(&cache.input.1, &cache.dct[0].1);
        let bg_0_to_1 = bind(&cache.dct[0].1, &cache.dct[1].1);
        let bg_1_to_0 = bind(&cache.dct[1].1, &cache.dct[0].1);
        // Ping pong buffer
        let bind_groups = [&bg_to_0, &bg_0_to_1, &bg_1_to_0, &bg_0_to_1, &bg_1_to_0];

        for (pipeline, bind_group) in self.pipelines.iter().zip(bind_groups) {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_immediates(0, bytemuck::bytes_of(&params));
            pass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
        }

        let row_stride = (width * 4).next_multiple_of(256);
        let staging_size = row_stride as u64 * height as u64;

        let staging = device.create_buffer(&BufferDescriptor {
            label: None,
            size: staging_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &cache.output.0,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &staging,
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

        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        rx.recv().unwrap().unwrap();

        {
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
        }
    }

    fn ensure_cache(&mut self, device: &Device, w: u32, h: u32) {
        if self.cache.as_ref().is_some_and(|c| c.w == w && c.h == h) {
            return;
        }

        let make_tex = |format: TextureFormat, usage: TextureUsages| {
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
        };

        let input = make_tex(
            TextureFormat::Rgba8Unorm,
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        );
        let output = make_tex(
            TextureFormat::Rgba8Unorm,
            TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        );
        let dct = [
            make_tex(
                TextureFormat::Rgba16Float,
                TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            ),
            make_tex(
                TextureFormat::Rgba16Float,
                TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            ),
        ];
        let upload_buf = vec![0u8; (w * h * 4) as usize];

        self.cache = Some(Cache {
            w,
            h,
            input,
            output,
            dct,
            upload_buf,
        });
    }
}
