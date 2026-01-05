use crate::types::ParamIdx;
use after_effects::{self as ae, Error};

pub const INPUT_LAYER_CHECKOUT_ID: i32 = 100;
pub const MATTE_LAYER_CHECKOUT_ID: i32 = 201;

pub fn setup_params(params: &mut ae::Parameters<ParamIdx>) -> Result<(), Error> {
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

    params.add(
        ParamIdx::CoefficientMin,
        "Coefficient Min",
        ae::FloatSliderDef::setup(|f| {
            f.set_default(0.0);
            f.set_valid_min(0.0);
            f.set_valid_max(1.0);
            f.set_slider_min(0.0);
            f.set_slider_max(0.5);
            f.set_precision(3);
        }),
    )?;

    params.add(
        ParamIdx::CoefficientMax,
        "Coefficient Max",
        ae::FloatSliderDef::setup(|f| {
            f.set_default(1.0);
            f.set_valid_min(0.0);
            f.set_valid_max(1.0);
            f.set_slider_min(0.0);
            f.set_slider_max(0.5);
            f.set_precision(3);
        }),
    )?;

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

    params.add(
        ParamIdx::ColorSpace,
        "Color Space",
        ae::PopupDef::setup(|f| {
            f.set_options(&["RGB", "YCbCr (JPEG)"]);
            f.set_default(2); // YCbCr default (1-indexed)
        }),
    )?;

    params.add(
        ParamIdx::LumaQualityMatte,
        "Luma Quality Matte",
        ae::LayerDef::setup(|_| {}),
    )?;

    Ok(())
}
