use crate::types::ParamIdx;
use after_effects::{self as ae, Error};

pub const INPUT_LAYER_CHECKOUT_ID: i32 = 100;

pub fn setup_params(params: &mut ae::Parameters<ParamIdx>) -> Result<(), Error> {
    // Quality (1-100)
    params.add(
        ParamIdx::Quality,
        "Quality",
        ae::FloatSliderDef::setup(|f| {
            f.set_default(50.0);
            f.set_valid_min(1.0);
            f.set_valid_max(100.0);
            f.set_slider_min(1.0);
            f.set_slider_max(100.0);
            f.set_precision(1);
        }),
    )?;

    // Block Size (2-64)
    params.add(
        ParamIdx::BlockSize,
        "Block Size",
        ae::SliderDef::setup(|f| {
            f.set_default(8);
            f.set_valid_min(2);
            f.set_valid_max(64);
            f.set_slider_min(2);
            f.set_slider_max(64);
        }),
    )?;

    // Coefficient Threshold (0-1)
    params.add(
        ParamIdx::CoefficientThreshold,
        "Coefficient Threshold",
        ae::FloatSliderDef::setup(|f| {
            f.set_default(0.0);
            f.set_valid_min(0.0);
            f.set_valid_max(1.0);
            f.set_slider_min(0.0);
            f.set_slider_max(1.0);
            f.set_precision(2);
        }),
    )?;

    // Blend Original (0-1)
    params.add(
        ParamIdx::BlendOriginal,
        "Blend Original",
        ae::FloatSliderDef::setup(|f| {
            f.set_default(0.0);
            f.set_valid_min(0.0);
            f.set_valid_max(1.0);
            f.set_slider_min(0.0);
            f.set_slider_max(1.0);
            f.set_precision(2);
        }),
    )?;

    // Error Rate (0-100)
    params.add(
        ParamIdx::ErrorRate,
        "Error Rate",
        ae::FloatSliderDef::setup(|f| {
            f.set_default(0.0);
            f.set_valid_min(0.0);
            f.set_valid_max(100.0);
            f.set_slider_min(0.0);
            f.set_slider_max(100.0);
            f.set_precision(1);
        }),
    )?;

    // Error Brightness Min (-1 to 1)
    params.add(
        ParamIdx::ErrorBrightnessMin,
        "Error Brightness Min",
        ae::FloatSliderDef::setup(|f| {
            f.set_default(0.0);
            f.set_valid_min(-1.0);
            f.set_valid_max(1.0);
            f.set_slider_min(-1.0);
            f.set_slider_max(1.0);
            f.set_precision(2);
        }),
    )?;

    // Error Brightness Max (-1 to 1)
    params.add(
        ParamIdx::ErrorBrightnessMax,
        "Error Brightness Max",
        ae::FloatSliderDef::setup(|f| {
            f.set_default(0.0);
            f.set_valid_min(-1.0);
            f.set_valid_max(1.0);
            f.set_slider_min(-1.0);
            f.set_slider_max(1.0);
            f.set_precision(2);
        }),
    )?;

    // Error Blue Yellow Min (-1 to 1)
    params.add(
        ParamIdx::ErrorBlueYellowMin,
        "Error Blue Yellow Min",
        ae::FloatSliderDef::setup(|f| {
            f.set_default(0.0);
            f.set_valid_min(-1.0);
            f.set_valid_max(1.0);
            f.set_slider_min(-1.0);
            f.set_slider_max(1.0);
            f.set_precision(2);
        }),
    )?;

    // Error Blue Yellow Max (-1 to 1)
    params.add(
        ParamIdx::ErrorBlueYellowMax,
        "Error Blue Yellow Max",
        ae::FloatSliderDef::setup(|f| {
            f.set_default(0.0);
            f.set_valid_min(-1.0);
            f.set_valid_max(1.0);
            f.set_slider_min(-1.0);
            f.set_slider_max(1.0);
            f.set_precision(2);
        }),
    )?;

    // Error Red Cyan Min (-1 to 1)
    params.add(
        ParamIdx::ErrorRedCyanMin,
        "Error Red Cyan Min",
        ae::FloatSliderDef::setup(|f| {
            f.set_default(0.0);
            f.set_valid_min(-1.0);
            f.set_valid_max(1.0);
            f.set_slider_min(-1.0);
            f.set_slider_max(1.0);
            f.set_precision(2);
        }),
    )?;

    // Error Red Cyan Max (-1 to 1)
    params.add(
        ParamIdx::ErrorRedCyanMax,
        "Error Red Cyan Max",
        ae::FloatSliderDef::setup(|f| {
            f.set_default(0.0);
            f.set_valid_min(-1.0);
            f.set_valid_max(1.0);
            f.set_slider_min(-1.0);
            f.set_slider_max(1.0);
            f.set_precision(2);
        }),
    )?;

    // Seed (0-10000)
    params.add(
        ParamIdx::Seed,
        "Seed",
        ae::SliderDef::setup(|f| {
            f.set_default(0);
            f.set_valid_min(0);
            f.set_valid_max(10000);
            f.set_slider_min(0);
            f.set_slider_max(10000);
        }),
    )?;

    // Error Matte Mode (dropdown)
    params.add(
        ParamIdx::ErrorMatteMode,
        "Error Matte Mode",
        ae::PopupDef::setup(|f| {
            f.set_options(&["Luminance", "RGB Drive YCbCr"]);
            f.set_default(1);
        }),
    )?;

    // Error Matte (layer)
    params.add(
        ParamIdx::ErrorMatte,
        "Error Matte",
        ae::LayerDef::setup(|_| {}),
    )?;

    // Luma Quality Matte (layer)
    params.add(
        ParamIdx::LumaQualityMatte,
        "Luma Quality Matte",
        ae::LayerDef::setup(|_| {}),
    )?;

    Ok(())
}
