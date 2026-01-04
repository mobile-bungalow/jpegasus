use crate::param_util::INPUT_LAYER_CHECKOUT_ID;
use crate::pipeline::{DctPushConstants, Layer};
use crate::types::*;

use ae::*;
use after_effects as ae;

pub fn render(
    state: &mut super::PluginState,
    local: &mut Local,
    extra: &SmartRenderExtra,
) -> Result<(), after_effects::Error> {
    let params = load_parameters(state)?;

    let Some(global) = state.global.get() else {
        return Err(Error::Generic);
    };

    let cb = extra.callbacks();

    let Some(input_layer) = cb.checkout_layer_pixels(INPUT_LAYER_CHECKOUT_ID as u32)? else {
        return Ok(());
    };

    let width = input_layer.width() as u32;
    let height = input_layer.height() as u32;
    let input_row_bytes = input_layer.row_bytes().unsigned_abs();

    let luma_quality_layer = cb
        .checkout_layer_pixels(ParamIdx::LumaQualityMatte.idx() as u32)
        .ok()
        .flatten();

    let mut push_constants = params;
    push_constants.use_luma_quality = if luma_quality_layer.is_some() { 1 } else { 0 };

    let Some(mut out_layer) = cb.checkout_output()? else {
        return Ok(());
    };
    let output_row_bytes = out_layer.row_bytes().unsigned_abs();
    let bit_depth = BitDepth::from(extra.bit_depth());

    let luma_quality = luma_quality_layer.as_ref().map(|l| Layer {
        buffer: l.buffer(),
        row_bytes: l.row_bytes().unsigned_abs(),
    });

    let pipeline = local.pipeline(&global.device);
    pipeline.render(
        &global.device,
        &global.queue,
        push_constants,
        input_layer.buffer(),
        out_layer.buffer_mut(),
        width,
        height,
        input_row_bytes,
        output_row_bytes,
        bit_depth,
        None,
        luma_quality,
    );

    Ok(())
}

macro_rules! checkout {
    ($in_data:expr, $time:expr, $idx:expr, float) => {
        ParamDef::checkout($in_data, $idx.idx(), $time.0, $time.1, $time.2, None)?
            .as_float_slider()?
            .value() as f32
    };
    ($in_data:expr, $time:expr, $idx:expr, int) => {
        ParamDef::checkout($in_data, $idx.idx(), $time.0, $time.1, $time.2, None)?
            .as_slider()?
            .value() as u32
    };
    ($in_data:expr, $time:expr, $idx:expr, popup) => {
        ParamDef::checkout($in_data, $idx.idx(), $time.0, $time.1, $time.2, None)?
            .as_popup()?
            .value() as u32
            - 1
    };
}

fn load_parameters(state: &super::PluginState) -> Result<DctPushConstants, after_effects::Error> {
    let in_data = state.in_data;
    let time = (
        in_data.current_time(),
        in_data.time_step(),
        in_data.time_scale(),
    );

    let quality = checkout!(in_data, time, ParamIdx::Quality, float);
    let block_size = checkout!(in_data, time, ParamIdx::BlockSize, int);
    let coefficient_min = checkout!(in_data, time, ParamIdx::CoefficientMin, float);
    let coefficient_max = checkout!(in_data, time, ParamIdx::CoefficientMax, float);
    let blend_original = checkout!(in_data, time, ParamIdx::BlendOriginal, float);
    let chroma_subsampling = checkout!(in_data, time, ParamIdx::ChromaSubsampling, popup);

    let mut params = DctPushConstants {
        block_size,
        coefficient_min,
        coefficient_max,
        blend_original,
        chroma_subsampling,
        ae_channel_order: 1,
        ..DctPushConstants::new()
    };
    params.set_quality(quality);
    Ok(params)
}
