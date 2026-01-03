use crate::u16_conversion::*;
use std::sync::Mutex;
use strum::{EnumCount, EnumIter, FromRepr};
use tweak_shader::wgpu::{self, Device, Queue};

// Hardcoded shader source
pub const DCT_SHADER: &str = include_str!("./resources/dct.glsl");

// Parameter indices matching dct.glsl inputs (starts at 1 for AE)
#[repr(u8)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Copy, Hash, EnumIter, EnumCount, FromRepr)]
pub enum ParamIdx {
    Quality = 1,
    BlockSize,
    CoefficientThreshold,
    BlendOriginal,
    ErrorRate,
    ErrorBrightnessMin,
    ErrorBrightnessMax,
    ErrorBlueYellowMin,
    ErrorBlueYellowMax,
    ErrorRedCyanMin,
    ErrorRedCyanMax,
    Seed,
    ErrorMatteMode,
    ErrorMatte,
    LumaQualityMatte,
}

impl ParamIdx {
    pub const fn idx(&self) -> i32 {
        *self as i32
    }
}

impl From<u8> for ParamIdx {
    fn from(value: u8) -> Self {
        Self::from_repr(value).unwrap_or(ParamIdx::Quality)
    }
}

impl From<ParamIdx> for u8 {
    fn from(value: ParamIdx) -> Self {
        value as u8
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(i16)]
pub enum BitDepth {
    U8 = 8,
    U16 = 16,
    F32 = 32,
    Invalid(i16),
}

impl From<i16> for BitDepth {
    fn from(value: i16) -> Self {
        match value {
            8 => BitDepth::U8,
            16 => BitDepth::U16,
            32 => BitDepth::F32,
            v => BitDepth::Invalid(v),
        }
    }
}

impl TryFrom<BitDepth> for wgpu::TextureFormat {
    type Error = i16;
    fn try_from(value: BitDepth) -> Result<wgpu::TextureFormat, Self::Error> {
        match value {
            BitDepth::U8 => Ok(wgpu::TextureFormat::Rgba8Unorm),
            BitDepth::U16 => Ok(wgpu::TextureFormat::Rgba16Float),
            BitDepth::F32 => Ok(wgpu::TextureFormat::Rgba32Float),
            BitDepth::Invalid(v) => Err(v),
        }
    }
}

impl From<wgpu::TextureFormat> for BitDepth {
    fn from(value: wgpu::TextureFormat) -> Self {
        match value {
            wgpu::TextureFormat::Rgba8Unorm => BitDepth::U8,
            wgpu::TextureFormat::Rgba16Float => BitDepth::U16,
            wgpu::TextureFormat::Rgba32Float => BitDepth::F32,
            _ => BitDepth::Invalid(-42),
        }
    }
}

pub enum JpegasusGlobal {
    Init(InnerGlobal),
    Uninit,
}

impl JpegasusGlobal {
    pub fn as_init(&self) -> Option<&InnerGlobal> {
        match self {
            Self::Init(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct InnerGlobal {
    pub device: Device,
    pub queue: Queue,
}

pub type LocalMutex = Mutex<Local>;

#[derive(Debug, Default)]
pub struct Local {
    pub local_init: Option<LocalInit>,
}

#[derive(Debug)]
pub struct LocalInit {
    pub ctx: tweak_shader::RenderContext,
    pub fmt: wgpu::TextureFormat,
    pub u16_converter: Option<U16ConversionContext>,
}

impl Default for JpegasusGlobal {
    fn default() -> Self {
        let instance_desc = wgpu::InstanceDescriptor {
            #[cfg(target_os = "windows")]
            backends: wgpu::Backends::VULKAN,
            #[cfg(target_os = "macos")]
            backends: wgpu::Backends::METAL,
            ..Default::default()
        };

        let instance = wgpu::Instance::new(instance_desc);

        let maybe_adapter = pollster::block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
        });

        let Some(adapter) = maybe_adapter else {
            return Self::Uninit;
        };

        let adapter_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::default().using_resolution(adapter_limits.clone());

        required_limits.max_push_constant_size = 256;
        required_limits.max_storage_textures_per_shader_stage = 4;
        required_limits.max_buffer_size = adapter_limits.max_buffer_size;
        required_limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;
        required_limits.max_texture_dimension_2d = adapter_limits.max_texture_dimension_2d;

        let maybe_dq = pollster::block_on(async {
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        required_features: wgpu::Features::PUSH_CONSTANTS
                            | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
                            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                            | wgpu::Features::VERTEX_WRITABLE_STORAGE,
                        required_limits,
                    },
                    None,
                )
                .await
        });

        let (device, queue) = match maybe_dq {
            Err(_) => return Self::Uninit,
            Ok((device, queue)) => (device, queue),
        };

        device.on_uncaptured_error(Box::new(|e| match e {
            wgpu::Error::Internal {
                source,
                description,
            } => {
                panic!("Internal GPU Error! {source} : {description}");
            }
            wgpu::Error::OutOfMemory { .. } => {
                panic!("Out of memory");
            }
            wgpu::Error::Validation {
                description,
                source,
            } => {
                panic!("{description} : {source}");
            }
        }));

        JpegasusGlobal::Init(InnerGlobal { device, queue })
    }
}

impl LocalInit {
    pub fn new(device: &Device, queue: &Queue, fmt: wgpu::TextureFormat) -> Self {
        let ctx = tweak_shader::RenderContext::new(DCT_SHADER, fmt, device, queue)
            .expect("Failed to compile embedded dct.glsl shader");

        let u16_converter = if fmt == wgpu::TextureFormat::Rgba16Float {
            Some(U16ConversionContext::new(device, queue))
        } else {
            None
        };

        LocalInit {
            ctx,
            fmt,
            u16_converter,
        }
    }
}

impl Local {
    pub fn init_or_update(&mut self, device: &Device, queue: &Queue, bit_depth: BitDepth) {
        let expected_fmt: wgpu::TextureFormat = bit_depth
            .try_into()
            .unwrap_or(wgpu::TextureFormat::Rgba8Unorm);

        match &self.local_init {
            None => {
                self.local_init = Some(LocalInit::new(device, queue, expected_fmt));
            }
            Some(LocalInit { fmt, .. }) => {
                if *fmt != expected_fmt {
                    self.local_init = Some(LocalInit::new(device, queue, expected_fmt));
                }
            }
        }
    }
}
