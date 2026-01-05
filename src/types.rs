use crate::pipeline::DctPipeline;
use std::sync::{Mutex, OnceLock};
use strum::{EnumCount, EnumIter, FromRepr};
use wgpu::{Device, Queue};

#[allow(dead_code)]
static FATAL_GPU_ERROR: OnceLock<Mutex<Option<String>>> = OnceLock::new();

#[allow(dead_code)]
fn error_store() -> &'static Mutex<Option<String>> {
    FATAL_GPU_ERROR.get_or_init(|| Mutex::new(None))
}

#[allow(dead_code)]
pub fn set_gpu_error(error: String) {
    if let Ok(mut guard) = error_store().lock() {
        *guard = Some(error);
    }
}

#[allow(dead_code)]
pub fn take_gpu_error() -> Option<String> {
    error_store().lock().ok().and_then(|mut g| g.take())
}

// Parameter indices matching dct.glsl inputs (starts at 1 for AE)
#[repr(u8)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Copy, Hash, EnumIter, EnumCount, FromRepr)]
pub enum ParamIdx {
    Quality = 1,
    BlockSize,
    CoefficientMin,
    CoefficientMax,
    BlendOriginal,
    ColorSpace,

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

impl BitDepth {
    pub const fn bytes_per_pixel(&self) -> usize {
        match self {
            BitDepth::U8 => 4,
            BitDepth::U16 => 8,
            BitDepth::F32 => 16,
            BitDepth::Invalid(_) => 4,
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

#[derive(Debug)]
#[allow(dead_code)]
pub struct InnerGlobal {
    pub device: Device,
    pub queue: Queue,
    pub pipeline: DctPipeline,
}

#[allow(dead_code)]
pub type JpegasusGlobal = std::sync::OnceLock<Mutex<InnerGlobal>>;

#[allow(dead_code)]
pub type LocalMutex = Mutex<Local>;

#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct Local;

#[allow(dead_code)]
pub fn init_global() -> Option<InnerGlobal> {
    let instance_desc = wgpu::InstanceDescriptor {
        #[cfg(target_os = "windows")]
        backends: wgpu::Backends::VULKAN,
        #[cfg(target_os = "macos")]
        backends: wgpu::Backends::METAL,
        ..Default::default()
    };

    let instance = wgpu::Instance::new(&instance_desc);

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .ok()?;

    let adapter_limits = adapter.limits();
    let mut required_limits = wgpu::Limits::default().using_resolution(adapter_limits.clone());

    required_limits.max_immediate_size = 256;
    required_limits.max_storage_textures_per_shader_stage = 4;
    required_limits.max_buffer_size = adapter_limits.max_buffer_size;
    required_limits.max_storage_buffer_binding_size =
        adapter_limits.max_storage_buffer_binding_size;
    required_limits.max_texture_dimension_2d = adapter_limits.max_texture_dimension_2d;

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::VERTEX_WRITABLE_STORAGE
            | wgpu::Features::IMMEDIATES,
        required_limits,
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::default(),
    }))
    .ok()?;

    device.on_uncaptured_error(std::sync::Arc::new(|e| match e {
        wgpu::Error::Internal {
            source,
            description,
        } => {
            set_gpu_error(format!("Internal GPU Error: {description} ({source})"));
        }
        wgpu::Error::OutOfMemory { .. } => {
            set_gpu_error("GPU Out of Memory".to_string());
        }
        wgpu::Error::Validation {
            description,
            source,
        } => {
            set_gpu_error(format!("GPU Validation Error: {description} ({source})"));
        }
    }));

    let pipeline = DctPipeline::new(&device);

    Some(InnerGlobal {
        device,
        queue,
        pipeline,
    })
}
