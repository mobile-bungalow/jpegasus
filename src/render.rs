use tweak_shader::input_type::ShaderBool;
use tweak_shader::TextureDesc;

use super::*;
use crate::param_util::INPUT_LAYER_CHECKOUT_ID;

pub fn render(
    state: &mut super::PluginState,
    instance: &mut super::Local,
    extra: &SmartRenderExtra,
) -> Result<(), after_effects::Error> {
    let Some(global) = state.global.as_init() else {
        return Err(Error::Generic);
    };

    let Some(LocalInit {
        ref mut ctx,
        u16_converter,
        fmt,
        ..
    }) = instance.local_init.as_mut()
    else {
        return Err(Error::Generic);
    };

    // Load all parameters into the shader context
    load_parameters(ctx, state)?;

    // Collect layer inputs
    let cb = extra.callbacks();
    let mut layers: Vec<(&str, _)> = Vec::new();

    // Input image (current layer)
    if let Ok(Some(layer)) = cb.checkout_layer_pixels(INPUT_LAYER_CHECKOUT_ID as u32) {
        layers.push(("input_image", layer));
    }

    // Error matte
    if let Ok(Some(layer)) = cb.checkout_layer_pixels(ParamIdx::ErrorMatte.idx() as u32) {
        layers.push(("error_matte", layer));
        // Auto-set use_error_matte
        if let Some(mut input) = ctx.get_input_mut("use_error_matte") {
            if let Some(b) = input.as_bool() {
                b.current = ShaderBool::True;
            }
        }
    } else {
        ctx.remove_texture("error_matte");
        if let Some(mut input) = ctx.get_input_mut("use_error_matte") {
            if let Some(b) = input.as_bool() {
                b.current = ShaderBool::False;
            }
        }
    }

    // Luma quality matte
    if let Ok(Some(layer)) = cb.checkout_layer_pixels(ParamIdx::LumaQualityMatte.idx() as u32) {
        layers.push(("luma_quality_matte", layer));
        // Auto-set use_luma_quality
        if let Some(mut input) = ctx.get_input_mut("use_luma_quality") {
            if let Some(b) = input.as_bool() {
                b.current = ShaderBool::True;
            }
        }
    } else {
        ctx.remove_texture("luma_quality_matte");
        if let Some(mut input) = ctx.get_input_mut("use_luma_quality") {
            if let Some(b) = input.as_bool() {
                b.current = ShaderBool::False;
            }
        }
    }

    // Always set ae_channel_order to true
    if let Some(mut input) = ctx.get_input_mut("ae_channel_order") {
        if let Some(b) = input.as_bool() {
            b.current = ShaderBool::True;
        }
    }

    if let Some(converter) = u16_converter {
        converter.prepare_cpu_layer_inputs(&global.device, &global.queue, layers.into_iter());

        let Some(mut out_layer) = cb.checkout_output()? else {
            return Ok(());
        };

        ctx.update_resolution([out_layer.width() as f32, out_layer.height() as f32]);
        converter.render_u15_to_cpu_buffer(&mut out_layer, &global.device, &global.queue, ctx);
    } else {
        for (name, layer) in layers.iter() {
            let real_fmt = *fmt;
            ctx.load_texture(
                name,
                TextureDesc {
                    width: layer.width() as u32,
                    height: layer.height() as u32,
                    stride: Some(layer.buffer_stride() as u32),
                    data: layer.buffer(),
                    format: real_fmt,
                },
                &global.device,
                &global.queue,
            );
        }

        let Some(mut out_layer) = cb.checkout_output()? else {
            return Ok(());
        };

        let width = out_layer.width() as u32;
        let height = out_layer.height() as u32;
        let stride = out_layer.buffer_stride() as u32;

        let limits = global.device.limits();
        let buffer_size = stride as u64 * height as u64;
        if buffer_size > limits.max_buffer_size {
            state.out_data.set_error_msg(&format!(
                "Buffer size {} exceeds GPU max {}",
                buffer_size, limits.max_buffer_size
            ));
            return Ok(());
        }
        if width > limits.max_texture_dimension_2d || height > limits.max_texture_dimension_2d {
            state.out_data.set_error_msg(&format!(
                "Texture {}x{} exceeds GPU max {}",
                width, height, limits.max_texture_dimension_2d
            ));
            return Ok(());
        }

        ctx.update_resolution([width as f32, height as f32]);
        ctx.render_to_slice(
            &global.queue,
            &global.device,
            width,
            height,
            out_layer.buffer_mut(),
            Some(stride),
        );
    }

    Ok(())
}

fn load_parameters(
    ctx: &mut tweak_shader::RenderContext,
    state: &super::PluginState,
) -> Result<(), after_effects::Error> {
    let in_data = state.in_data;
    let current_time = in_data.current_time();
    let current_frame = in_data.current_frame();
    let current_delta = in_data.time_step();
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
    if let Some(mut input) = ctx.get_input_mut("quality") {
        if let Some(f) = input.as_float() {
            f.current = quality;
        }
    }

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
    .value();
    if let Some(mut input) = ctx.get_input_mut("block_size") {
        if let Some(i) = input.as_int() {
            i.value.current = block_size;
        }
    }

    // Coefficient Threshold
    let coef_threshold = ParamDef::checkout(
        in_data,
        ParamIdx::CoefficientThreshold.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;
    if let Some(mut input) = ctx.get_input_mut("coefficient_threshold") {
        if let Some(f) = input.as_float() {
            f.current = coef_threshold;
        }
    }

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
    if let Some(mut input) = ctx.get_input_mut("blend_original") {
        if let Some(f) = input.as_float() {
            f.current = blend_original;
        }
    }

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
    if let Some(mut input) = ctx.get_input_mut("error_rate") {
        if let Some(f) = input.as_float() {
            f.current = error_rate;
        }
    }

    // Error Brightness Min
    let val = ParamDef::checkout(
        in_data,
        ParamIdx::ErrorBrightnessMin.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;
    if let Some(mut input) = ctx.get_input_mut("error_brightness_min") {
        if let Some(f) = input.as_float() {
            f.current = val;
        }
    }

    // Error Brightness Max
    let val = ParamDef::checkout(
        in_data,
        ParamIdx::ErrorBrightnessMax.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;
    if let Some(mut input) = ctx.get_input_mut("error_brightness_max") {
        if let Some(f) = input.as_float() {
            f.current = val;
        }
    }

    // Error Blue Yellow Min
    let val = ParamDef::checkout(
        in_data,
        ParamIdx::ErrorBlueYellowMin.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;
    if let Some(mut input) = ctx.get_input_mut("error_blue_yellow_min") {
        if let Some(f) = input.as_float() {
            f.current = val;
        }
    }

    // Error Blue Yellow Max
    let val = ParamDef::checkout(
        in_data,
        ParamIdx::ErrorBlueYellowMax.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;
    if let Some(mut input) = ctx.get_input_mut("error_blue_yellow_max") {
        if let Some(f) = input.as_float() {
            f.current = val;
        }
    }

    // Error Red Cyan Min
    let val = ParamDef::checkout(
        in_data,
        ParamIdx::ErrorRedCyanMin.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;
    if let Some(mut input) = ctx.get_input_mut("error_red_cyan_min") {
        if let Some(f) = input.as_float() {
            f.current = val;
        }
    }

    // Error Red Cyan Max
    let val = ParamDef::checkout(
        in_data,
        ParamIdx::ErrorRedCyanMax.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_float_slider()?
    .value() as f32;
    if let Some(mut input) = ctx.get_input_mut("error_red_cyan_max") {
        if let Some(f) = input.as_float() {
            f.current = val;
        }
    }

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
    .value();
    if let Some(mut input) = ctx.get_input_mut("seed") {
        if let Some(i) = input.as_int() {
            i.value.current = seed;
        }
    }

    // Error Matte Mode
    let mode = ParamDef::checkout(
        in_data,
        ParamIdx::ErrorMatteMode.idx(),
        current_time,
        time_step,
        time_scale,
        None,
    )?
    .as_popup()?
    .value()
        - 1; // AE popups are 1-indexed
    if let Some(mut input) = ctx.get_input_mut("error_matte_mode") {
        if let Some(i) = input.as_int() {
            i.value.current = mode;
        }
    }

    // Update time/frame
    ctx.update_time(current_time as f32 / time_scale as f32);
    ctx.update_frame_count(current_frame as u32);
    ctx.update_delta(current_delta as f32);

    Ok(())
}
