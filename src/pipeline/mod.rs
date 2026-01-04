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
    pub pass_index: u32,
    pub block_size: u32,
    pub quantization_step: f32,
    pub coefficient_min: f32,
    pub coefficient_max: f32,
    pub blend_original: f32,
    pub use_luma_quality: u32,
    pub ae_channel_order: u32,
    pub chroma_subsampling: u32,
}

impl DctPushConstants {
    pub fn new() -> Self {
        Self {
            coefficient_max: 1.0,
            ..Zeroable::zeroed()
        }
    }
    pub fn set_quality(&mut self, quality: f32) {
        let c = (101.0 - quality) / 100.0;
        self.quantization_step = c * c * c * 2.0;
    }
}

pub struct DctPipeline {
    pipelines: [ComputePipeline; 9],
    layout: BindGroupLayout,
    dummy: TextureView,
    cache: Option<Cache>,
}

struct Cache {
    w: u32,
    h: u32,
    fmt: TextureFormat,
    input: (Texture, TextureView),
    output_f16: (Texture, TextureView),
    output_8: (Texture, TextureView),
    output_16: (Texture, TextureView),
    scratch: [(Texture, TextureView); 2],
    staging: Buffer,
}

impl std::fmt::Debug for DctPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DctPipeline").finish_non_exhaustive()
    }
}

impl DctPipeline {
    pub fn new(device: &Device) -> Self {
        let shader = device.create_shader_module(include_wgsl!("../resources/dct.wgsl"));

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                tex_entry(0),
                storage_entry(1, TextureFormat::Rgba16Float),
                tex_entry(2),
                tex_entry(3),
                tex_entry(4),
                storage_entry(5, TextureFormat::Rgba8Unorm),
                storage_entry(6, TextureFormat::Rgba16Unorm),
            ],
        });

        let pipe_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<DctPushConstants>() as u32,
            }],
        });

        let entries = [
            "pass_rgb_to_ycbcr",
            "pass_dct_rows",
            "pass_dct_cols",
            "pass_quantize",
            "pass_idct_cols",
            "pass_idct_rows",
            "pass_finalize",
            "pass_finalize_8bit",
            "pass_finalize_16bit",
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

        let dummy = device
            .create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            })
            .create_view(&TextureViewDescriptor::default());

        Self {
            pipelines,
            layout,
            dummy,
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

        let fmt = match bit_depth {
            BitDepth::U8 => TextureFormat::Rgba8Unorm,
            BitDepth::U16 => TextureFormat::Rgba16Unorm,
            _ => TextureFormat::Rgba32Float,
        };
        let bpp = bit_depth.bytes_per_pixel();

        self.ensure_cache(device, width, height, fmt);
        let c = self.cache.as_ref().unwrap();

        // Upload input
        let packed = pack_rows(input.buffer, width, height, input.row_bytes, bpp);
        queue.write_texture(
            ImageCopyTexture {
                texture: &c.input.0,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &packed,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * bpp as u32),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // Upload quality matte if provided
        let qmatte_view = if let Some(m) = quality_matte {
            let packed = pack_rows(m.buffer, width, height, m.row_bytes, bpp);
            let tex = make_tex(
                device,
                width,
                height,
                fmt,
                TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            );
            queue.write_texture(
                ImageCopyTexture {
                    texture: &tex,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                &packed,
                ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(width * bpp as u32),
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
            device
                .create_texture(&TextureDescriptor {
                    label: None,
                    size: Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba16Float,
                    usage: TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor::default())
        };

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        let wg = (width.div_ceil(8), height.div_ceil(8));

        let s0 = &c.scratch[0].1;
        let s1 = &c.scratch[1].1;
        let inp = &c.input.1;
        let out_f16 = &c.output_f16.1;
        let out_8 = &c.output_8.1;
        let out_16 = &c.output_16.1;
        let dummy = &self.dummy;

        let dispatch = |enc: &mut CommandEncoder,
                        idx: usize,
                        b0: &TextureView,
                        b1: &TextureView,
                        params: &DctPushConstants| {
            let bg = device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(b0),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(b1),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(inp),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::TextureView(dummy),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: BindingResource::TextureView(&qmatte_view),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: BindingResource::TextureView(out_8),
                    },
                    BindGroupEntry {
                        binding: 6,
                        resource: BindingResource::TextureView(out_16),
                    },
                ],
            });
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines[idx]);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(params));
            pass.dispatch_workgroups(wg.0, wg.1, 1);
        };

        dispatch(&mut enc, 0, inp, s0, &params); // RGB→YCbCr: input→s0
        dispatch(&mut enc, 1, s0, s1, &params); // DCT rows: s0→s1
        dispatch(&mut enc, 2, s1, s0, &params); // DCT cols: s1→s0
        dispatch(&mut enc, 3, s0, s1, &params); // Quantize: s0→s1
        dispatch(&mut enc, 4, s1, s0, &params); // IDCT cols: s1→s0
        dispatch(&mut enc, 5, s0, s1, &params); // IDCT rows: s0→s1

        // Finalize based on bit depth
        let (final_idx, out_tex, out_bpp) = match bit_depth {
            BitDepth::U8 => (7, &c.output_8.0, 4u32),
            BitDepth::U16 => (8, &c.output_16.0, 8u32),
            _ => (6, &c.output_f16.0, 8u32),
        };
        dispatch(&mut enc, final_idx, s1, out_f16, &params);

        // Copy to staging
        let padded = (width * out_bpp + 255) & !255;
        enc.copy_texture_to_buffer(
            ImageCopyTexture {
                texture: out_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            ImageCopyBuffer {
                buffer: &c.staging,
                layout: ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(Some(enc.finish()));

        // Readback
        let slice = c.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        device.poll(Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        copy_rows(
            &data,
            output.buffer,
            width,
            height,
            padded as usize,
            output.row_bytes,
            out_bpp as usize,
        );
        drop(data);
        c.staging.unmap();
    }

    fn ensure_cache(&mut self, device: &Device, w: u32, h: u32, fmt: TextureFormat) {
        if self
            .cache
            .as_ref()
            .is_some_and(|c| c.w == w && c.h == h && c.fmt == fmt)
        {
            return;
        }

        let input = make_tex_view(
            device,
            w,
            h,
            fmt,
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        );
        let output_f16 = make_tex_view(
            device,
            w,
            h,
            TextureFormat::Rgba16Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        );
        let output_8 = make_tex_view(
            device,
            w,
            h,
            TextureFormat::Rgba8Unorm,
            TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        );
        let output_16 = make_tex_view(
            device,
            w,
            h,
            TextureFormat::Rgba16Unorm,
            TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        );
        let scratch = [
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
        let staging_size = ((w * 8 + 255) & !255) as u64 * h as u64;
        let staging = device.create_buffer(&BufferDescriptor {
            label: None,
            size: staging_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        self.cache = Some(Cache {
            w,
            h,
            fmt,
            input,
            output_f16,
            output_8,
            output_16,
            scratch,
            staging,
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

fn make_tex(
    device: &Device,
    w: u32,
    h: u32,
    format: TextureFormat,
    usage: TextureUsages,
) -> Texture {
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
        usage,
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
    let t = make_tex(device, w, h, format, usage);
    let v = t.create_view(&TextureViewDescriptor::default());
    (t, v)
}

fn pack_rows(input: &[u8], w: u32, h: u32, src_stride: usize, bpp: usize) -> Vec<u8> {
    let dst_stride = w as usize * bpp;
    if src_stride == dst_stride {
        return input[..h as usize * dst_stride].to_vec();
    }
    let mut out = vec![0u8; h as usize * dst_stride];
    for y in 0..h as usize {
        out[y * dst_stride..(y + 1) * dst_stride]
            .copy_from_slice(&input[y * src_stride..y * src_stride + dst_stride]);
    }
    out
}

fn copy_rows(
    src: &[u8],
    dst: &mut [u8],
    w: u32,
    h: u32,
    src_stride: usize,
    dst_stride: usize,
    bpp: usize,
) {
    let row_len = w as usize * bpp;
    if src_stride == dst_stride {
        dst[..h as usize * dst_stride].copy_from_slice(&src[..h as usize * src_stride]);
    } else {
        for y in 0..h as usize {
            dst[y * dst_stride..y * dst_stride + row_len]
                .copy_from_slice(&src[y * src_stride..y * src_stride + row_len]);
        }
    }
}
