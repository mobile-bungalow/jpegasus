use crate::param_util::INPUT_LAYER_CHECKOUT_ID;
use crate::pipeline::DctPushConstants;
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

    let error_matte_layer = cb
        .checkout_layer_pixels(ParamIdx::ErrorMatte.idx() as u32)
        .ok()
        .flatten();
    let luma_quality_layer = cb
        .checkout_layer_pixels(ParamIdx::LumaQualityMatte.idx() as u32)
        .ok()
        .flatten();

    let mut push_constants = params;
    push_constants.use_error_matte = if error_matte_layer.is_some() { 1 } else { 0 };
    push_constants.use_luma_quality = if luma_quality_layer.is_some() { 1 } else { 0 };

    let Some(mut out_layer) = cb.checkout_output()? else {
        return Ok(());
    };
    let output_row_bytes = out_layer.row_bytes().unsigned_abs();

    let bit_depth = BitDepth::from(extra.bit_depth());

    // Get row_bytes for each matte layer (they may differ from input)
    let error_matte_row_bytes = error_matte_layer
        .as_ref()
        .map(|l| l.row_bytes().unsigned_abs());
    let luma_quality_row_bytes = luma_quality_layer
        .as_ref()
        .map(|l| l.row_bytes().unsigned_abs());

    // Get thread-local pipeline (lazily initialized)
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
        error_matte_layer.as_ref().map(|l| l.buffer()),
        error_matte_row_bytes,
        luma_quality_layer.as_ref().map(|l| l.buffer()),
        luma_quality_row_bytes,
    );

    Ok(())
}

fn load_parameters(state: &super::PluginState) -> Result<DctPushConstants, after_effects::Error> {
    let in_data = state.in_data;
    let current_time = in_data.current_time();
    let time_step = in_data.time_step();
    let time_scale = in_data.time_scale();

    // Quality
    let quality = ParamDef::checkout(
        in_data,
        ParamIdx::Quality.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;

    // Block Size
    let block_size = ParamDef::checkout(
        in_data,
        ParamIdx::BlockSize.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_slider()?
    .value() as u32;

    // Coefficient Threshold
    let coefficient_threshold = ParamDef::checkout(
        in_data,
        ParamIdx::CoefficientThreshold.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;

    // Blend Original
    let blend_original = ParamDef::checkout(
        in_data,
        ParamIdx::BlendOriginal.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;

    // Error Rate
    let error_rate = ParamDef::checkout(
        in_data,
        ParamIdx::ErrorRate.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;

    // Error Brightness Min
    let error_brightness_min = ParamDef::checkout(
        in_data,
        ParamIdx::ErrorBrightnessMin.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;

    // Error Brightness Max
    let error_brightness_max = ParamDef::checkout(
        in_data,
        ParamIdx::ErrorBrightnessMax.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;

    // Error Blue Yellow Min
    let error_blue_yellow_min = ParamDef::checkout(
        in_data,
        ParamIdx::ErrorBlueYellowMin.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;

    // Error Blue Yellow Max
    let error_blue_yellow_max = ParamDef::checkout(
        in_data,
        ParamIdx::ErrorBlueYellowMax.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;

    // Error Red Cyan Min
    let error_red_cyan_min = ParamDef::checkout(
        in_data,
        ParamIdx::ErrorRedCyanMin.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;

    // Error Red Cyan Max
    let error_red_cyan_max = ParamDef::checkout(
        in_data,
        ParamIdx::ErrorRedCyanMax.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;

    // Seed
    let seed = ParamDef::checkout(
        in_data,
        ParamIdx::Seed.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_slider()?
    .value() as u32;

    // Error Matte Mode
    let error_matte_mode = ParamDef::checkout(
        in_data,
        ParamIdx::ErrorMatteMode.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_popup()?
    .value() as u32
        - 1; // AE popups are 1-indexed

    // Chroma Subsampling
    let chroma_subsampling = ParamDef::checkout(
        in_data,
        ParamIdx::ChromaSubsampling.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_popup()?
    .value() as u32
        - 1; // AE popups are 1-indexed

    let mut params = DctPushConstants {
        width: 0,
        height: 0,
        pass_index: 0,
        block_size,
        quantization_step: 0.0,
        coefficient_threshold,
        blend_original,
        error_rate,
        error_brightness_min,
        error_brightness_max,
        error_blue_yellow_min,
        error_blue_yellow_max,
        error_red_cyan_min,
        error_red_cyan_max,
        seed,
        error_matte_mode,
        use_error_matte: 0,
        use_luma_quality: 0,
        ae_channel_order: 1,
        chroma_subsampling,
    };
    params.set_quality(quality);
    Ok(params)
}
