// the ae macro does som unsanitary stuff
#![allow(clippy::drop_non_drop, clippy::question_mark)]

mod param_util;
pub mod pipeline;
mod render;
mod types;

use std::sync::Mutex;

use ae::*;
use after_effects as ae;
use after_effects_sys as ae_sys;
use types::*;

use crate::param_util::INPUT_LAYER_CHECKOUT_ID;

static PLUGIN_ID: std::sync::OnceLock<i32> = std::sync::OnceLock::new();

ae::define_effect!(JpegasusGlobal, LocalMutex, ParamIdx);

impl AdobePluginInstance for LocalMutex {
    fn flatten(&self) -> Result<(u16, Vec<u8>), Error> {
        Ok((1, Vec::new()))
    }

    fn unflatten(_version: u16, _serialized: &[u8]) -> Result<Self, Error> {
        Ok(Mutex::new(Local::default()))
    }

    fn render(&self, _: &mut PluginState, _: &Layer, _: &mut Layer) -> Result<(), ae::Error> {
        Ok(())
    }

    fn do_dialog(&mut self, _: &mut PluginState) -> Result<(), ae::Error> {
        Ok(())
    }

    fn handle_command(&mut self, plugin: &mut PluginState, command: Command) -> Result<(), Error> {
        let PluginState { in_data, .. } = plugin;

        match command {
            Command::About => plugin.out_data.set_return_msg("Jpegasus - DCT-ish effect"),
            Command::SmartPreRender { mut extra } => {
                let mut req = extra.output_request();
                let cb = extra.callbacks();

                req.field = ae_sys::PF_Field_FRAME as i32;
                req.preserve_rgb_of_zero_alpha = 1;
                req.channel_mask = ae_sys::PF_ChannelMask_ARGB as i32;

                let current_time = in_data.current_time();
                let time_step = in_data.time_step();
                let time_scale = in_data.time_scale();

                // Checkout input layer (current layer as filter) first to get the rect
                if let Ok(width_test) = cb.checkout_layer(
                    0,
                    INPUT_LAYER_CHECKOUT_ID - 1,
                    &req,
                    current_time,
                    time_step,
                    time_scale,
                ) {
                    req.rect = width_test.max_result_rect;

                    let full_checkout = cb.checkout_layer(
                        0,
                        INPUT_LAYER_CHECKOUT_ID,
                        &req,
                        current_time,
                        time_step,
                        time_scale,
                    )?;

                    // Now checkout matte layers with the same rect
                    let _ = cb.checkout_layer(
                        ParamIdx::ErrorMatte.idx(),
                        ParamIdx::ErrorMatte.idx(),
                        &req,
                        current_time,
                        time_step,
                        time_scale,
                    );

                    let _ = cb.checkout_layer(
                        ParamIdx::LumaQualityMatte.idx(),
                        ParamIdx::LumaQualityMatte.idx(),
                        &req,
                        current_time,
                        time_step,
                        time_scale,
                    );

                    extra.set_result_rect(full_checkout.result_rect.into());
                    extra.set_max_result_rect(full_checkout.result_rect.into());
                    extra.set_returns_extra_pixels(true);
                }
            }
            Command::SmartRender { extra } => {
                if let Some(gpu_error) = take_gpu_error() {
                    plugin
                        .out_data
                        .set_return_msg(&format!("Jpegasus: {gpu_error}"));
                    return Err(ae::Error::Generic);
                }
                render::render(plugin, &mut self.lock().unwrap(), &extra)?;
                if let Some(gpu_error) = take_gpu_error() {
                    plugin
                        .out_data
                        .set_return_msg(&format!("Jpegasus: {gpu_error}"));
                    return Err(ae::Error::Generic);
                }
            }
            _ => {}
        };

        Ok(())
    }
}

impl AdobePluginGlobal for JpegasusGlobal {
    fn params_setup(
        &self,
        params: &mut ae::Parameters<ParamIdx>,
        _in_data: InData,
        _out_data: OutData,
    ) -> Result<(), Error> {
        param_util::setup_params(params)?;
        Ok(())
    }

    fn handle_command(
        &mut self,
        cmd: ae::Command,
        _in_data: ae::InData,
        mut out_data: ae::OutData,
        _params: &mut ae::Parameters<ParamIdx>,
    ) -> Result<(), ae::Error> {
        match cmd {
            ae::Command::About => {
                out_data.set_return_msg("Jpegasus DCT JPEGGish effect.");
            }
            Command::GlobalSetup => {
                let suite = ae::aegp::suites::Utility::new()?;

                PLUGIN_ID
                    .set(suite.register_with_aegp("jpegasus")?)
                    .expect("already set");

                if self.get().is_none() {
                    if let Some(inner) = init_global() {
                        let _ = self.set(inner);
                    } else {
                        out_data.set_return_msg("Jpegasus failed to initialize GPU");
                        return Err(ae::Error::Generic);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}
