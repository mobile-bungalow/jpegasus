use crate::types::ParamIdx;
use after_effects::{self as ae, Error};

pub const INPUT_LAYER_CHECKOUT_ID: i32 = 100;

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

    params.add_group(
        ParamIdx::ErrorGroupStart,
        ParamIdx::ErrorGroupEnd,
        "Error",
        true,
        |params| {
            params.add(
                ParamIdx::ErrorRate,
                "Rate",
                ae::FloatSliderDef::setup(|f| {
                    f.set_default(0.0);
                    f.set_valid_min(0.0);
                    f.set_valid_max(100.0);
                    f.set_slider_min(0.0);
                    f.set_slider_max(100.0);
                    f.set_precision(1);
                }),
            )?;

            params.add(
                ParamIdx::ErrorBrightnessMin,
                "Brightness Min",
                ae::FloatSliderDef::setup(|f| {
                    f.set_default(0.0);
                    f.set_valid_min(-1.0);
                    f.set_valid_max(1.0);
                    f.set_slider_min(-1.0);
                    f.set_slider_max(1.0);
                    f.set_precision(2);
                }),
            )?;

            params.add(
                ParamIdx::ErrorBrightnessMax,
                "Brightness Max",
                ae::FloatSliderDef::setup(|f| {
                    f.set_default(0.0);
                    f.set_valid_min(-1.0);
                    f.set_valid_max(1.0);
                    f.set_slider_min(-1.0);
                    f.set_slider_max(1.0);
                    f.set_precision(2);
                }),
            )?;

            params.add(
                ParamIdx::ErrorBlueYellowMin,
                "Blue Yellow Min",
                ae::FloatSliderDef::setup(|f| {
                    f.set_default(0.0);
                    f.set_valid_min(-1.0);
                    f.set_valid_max(1.0);
                    f.set_slider_min(-1.0);
                    f.set_slider_max(1.0);
                    f.set_precision(2);
                }),
            )?;

            params.add(
                ParamIdx::ErrorBlueYellowMax,
                "Blue Yellow Max",
                ae::FloatSliderDef::setup(|f| {
                    f.set_default(0.0);
                    f.set_valid_min(-1.0);
                    f.set_valid_max(1.0);
                    f.set_slider_min(-1.0);
                    f.set_slider_max(1.0);
                    f.set_precision(2);
                }),
            )?;

            params.add(
                ParamIdx::ErrorRedCyanMin,
                "Red Cyan Min",
                ae::FloatSliderDef::setup(|f| {
                    f.set_default(0.0);
                    f.set_valid_min(-1.0);
                    f.set_valid_max(1.0);
                    f.set_slider_min(-1.0);
                    f.set_slider_max(1.0);
                    f.set_precision(2);
                }),
            )?;

            params.add(
                ParamIdx::ErrorRedCyanMax,
                "Red Cyan Max",
                ae::FloatSliderDef::setup(|f| {
                    f.set_default(0.0);
                    f.set_valid_min(-1.0);
                    f.set_valid_max(1.0);
                    f.set_slider_min(-1.0);
                    f.set_slider_max(1.0);
                    f.set_precision(2);
                }),
            )?;

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

            params.add(
                ParamIdx::ErrorMatteMode,
                "Matte Mode",
                ae::PopupDef::setup(|f| {
                    f.set_options(&["Luminance", "RGB Drive YCbCr"]);
                    f.set_default(1);
                }),
            )?;

            params.add(ParamIdx::ErrorMatte, "Matte", ae::LayerDef::setup(|_| {}))?;

            Ok(())
        },
    )?;

    params.add(
        ParamIdx::LumaQualityMatte,
        "Luma Quality Matte",
        ae::LayerDef::setup(|_| {}),
    )?;

    params.add(
        ParamIdx::ChromaSubsampling,
        "Chroma Subsampling",
        ae::PopupDef::setup(|f| {
            f.set_options(&["None (4:4:4)", "4:2:2", "4:2:0", "4:1:1"]);
            f.set_default(1);
        }),
    )?;

    Ok(())
}
