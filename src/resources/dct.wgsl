struct Params {
    width: u32,
    height: u32,
    pass_index: u32,
    block_size: u32,
    quantization_step: f32,
    coefficient_threshold: f32,
    blend_original: f32,
    error_rate: f32,
    error_brightness_min: f32,
    error_brightness_max: f32,
    error_blue_yellow_min: f32,
    error_blue_yellow_max: f32,
    error_red_cyan_min: f32,
    error_red_cyan_max: f32,
    seed: u32,
    error_matte_mode: u32,
    use_error_matte: u32,
    use_luma_quality: u32,
    ae_channel_order: u32,
    chroma_subsampling: u32,
}

var<push_constant> params: Params;

const PI: f32 = 3.14159265359;
const SQRT_HALF: f32 = 0.707106781;
const WORKGROUP_SIZE: u32 = 16u;
const MAX_BLOCK_SIZE: u32 = 64u;

var<workgroup> block: array<array<vec4<f32>, MAX_BLOCK_SIZE>, MAX_BLOCK_SIZE>;

@group(0) @binding(0) var input_image: texture_2d<f32>;
@group(0) @binding(1) var output_image: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var error_matte: texture_2d<f32>;
@group(0) @binding(3) var luma_quality_matte: texture_2d<f32>;

fn swizzle_in(c: vec4<f32>) -> vec4<f32> {
    if params.ae_channel_order == 1u {
        return c.gbar;
    }
    return c;
}

fn swizzle_out(c: vec4<f32>) -> vec4<f32> {
    if params.ae_channel_order == 1u {
        return c.argb;
    }
    return c;
}

fn hash(p: vec2<f32>, s: i32) -> f32 {
    var p3 = fract(vec3<f32>(p.x + f32(s), p.y + f32(s), p.x + f32(s)) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn dct_coeff(k: u32) -> f32 {
    if k == 0u {
        return SQRT_HALF;
    }
    return 1.0;
}

fn dct_basis(n: u32, k: u32, block_size: u32) -> f32 {
    return cos((2.0 * f32(n) + 1.0) * f32(k) * PI / (2.0 * f32(block_size)));
}

fn rgb_to_ycbcr(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b,
        -0.169 * rgb.r - 0.331 * rgb.g + 0.500 * rgb.b,
        0.500 * rgb.r - 0.419 * rgb.g - 0.081 * rgb.b
    );
}

fn ycbcr_to_rgb(ycbcr: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        ycbcr.x + 1.402 * ycbcr.z,
        ycbcr.x - 0.344 * ycbcr.y - 0.714 * ycbcr.z,
        ycbcr.x + 1.772 * ycbcr.y
    );
}

fn sample_error_matte(px: vec2<i32>) -> vec3<f32> {
    let matte = swizzle_in(textureLoad(error_matte, px, 0));
    let lum = 0.299 * matte.r + 0.587 * matte.g + 0.114 * matte.b;
    let use_lum = f32(1u - params.error_matte_mode);
    return mix(matte.rgb, vec3<f32>(lum), use_lum);
}

fn sample_luma_quality(px: vec2<i32>) -> f32 {
    let matte = swizzle_in(textureLoad(luma_quality_matte, px, 0));
    return 0.299 * matte.r + 0.587 * matte.g + 0.114 * matte.b;
}

fn apply_error(rgb: vec3<f32>, px: vec2<i32>, block_origin: vec2<i32>) -> vec3<f32> {
    var score: vec3<f32>;
    if params.use_error_matte == 0u {
        let block_hash = hash(vec2<f32>(f32(block_origin.x), f32(block_origin.y)), i32(params.seed));
        if block_hash * 100.0 >= params.error_rate {
            return rgb;
        }
        score = vec3<f32>(hash(vec2<f32>(f32(px.x), f32(px.y)), i32(params.seed)));
    } else {
        score = sample_error_matte(px);
    }

    var ycbcr = rgb_to_ycbcr(rgb);
    let err_min = vec3<f32>(params.error_brightness_min, params.error_blue_yellow_min, params.error_red_cyan_min);
    let err_max = vec3<f32>(params.error_brightness_max, params.error_blue_yellow_max, params.error_red_cyan_max);
    let rand = vec3<f32>(
        hash(vec2<f32>(f32(px.x), f32(px.y)), i32(params.seed) + 1),
        hash(vec2<f32>(f32(px.x), f32(px.y)), i32(params.seed) + 2),
        hash(vec2<f32>(f32(px.x), f32(px.y)), i32(params.seed) + 3)
    );
    let err = mix(err_min, err_max, rand) * score;
    ycbcr = ycbcr * (1.0 + err * 4.0);
    return ycbcr_to_rgb(ycbcr);
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn dct_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let bs = params.block_size;
    let block_origin = vec2<i32>(i32(wid.x * bs), i32(wid.y * bs));

    // DCT normalization factor: sqrt(2.0 / block_size)
    let dct_norm = sqrt(2.0 / f32(bs));

    // Each thread loops to cover the full block (bs x bs) in WORKGROUP_SIZE x WORKGROUP_SIZE tiles
    // Load input into shared memory, convert to YCbCr (centered around 0)
    for (var ty = lid.y; ty < bs; ty = ty + WORKGROUP_SIZE) {
        for (var tx = lid.x; tx < bs; tx = tx + WORKGROUP_SIZE) {
            let px = vec2<i32>(block_origin.x + i32(tx), block_origin.y + i32(ty));
            let in_bounds = px.x < i32(params.width) && px.y < i32(params.height);
            if in_bounds {
                let rgba = swizzle_in(textureLoad(input_image, px, 0));
                let ycbcr = rgb_to_ycbcr(rgba.rgb);
                block[ty][tx] = vec4<f32>(ycbcr.x - 0.5, ycbcr.y, ycbcr.z, rgba.a - 0.5);
            } else {
                block[ty][tx] = vec4<f32>(0.0);
            }
        }
    }
    workgroupBarrier();

    // Forward DCT on rows
    for (var ly = lid.y; ly < bs; ly = ly + WORKGROUP_SIZE) {
        for (var lx = lid.x; lx < bs; lx = lx + WORKGROUP_SIZE) {
            var row_sum = vec4<f32>(0.0);
            for (var n = 0u; n < bs; n = n + 1u) {
                row_sum = row_sum + block[ly][n] * dct_basis(n, lx, bs);
            }
            row_sum = row_sum * dct_norm * dct_coeff(lx);
            block[ly][lx] = row_sum;
        }
    }
    workgroupBarrier();

    // Forward DCT on columns, then quantize
    for (var ly = lid.y; ly < bs; ly = ly + WORKGROUP_SIZE) {
        for (var lx = lid.x; lx < bs; lx = lx + WORKGROUP_SIZE) {
            var col_sum = vec4<f32>(0.0);
            for (var n = 0u; n < bs; n = n + 1u) {
                col_sum = col_sum + block[n][lx] * dct_basis(n, ly, bs);
            }
            col_sum = col_sum * dct_norm * dct_coeff(ly);

            // Coefficient thresholding
            let freq = lx + ly;
            let max_freq = bs * 2u - 2u;
            let threshold = u32(f32(max_freq) * (1.0 - params.coefficient_threshold));
            if freq > threshold {
                col_sum = vec4<f32>(0.0);
            }

            // Quality-based quantization
            let px = vec2<i32>(block_origin.x + i32(lx), block_origin.y + i32(ly));
            let in_bounds = px.x < i32(params.width) && px.y < i32(params.height);
            var q = params.quantization_step;
            if params.use_luma_quality == 1u && in_bounds {
                let luma_q = sample_luma_quality(px);
                let compression = (101.0 - luma_q * 100.0) / 100.0;
                q = compression * compression * compression * 2.0;
            }

            // Chroma subsampling - apply stronger quantization to chroma channels
            let mode = params.chroma_subsampling;
            let chroma_q_mult = select(1.0, select(2.0, select(4.0, 8.0, mode == 3u), mode >= 2u), mode >= 1u);
            let chroma_q = q * chroma_q_mult;

            // Quantize luma (Y) and alpha with base quantization
            col_sum.x = select(col_sum.x, floor(col_sum.x / q + 0.5) * q, q > 0.0001);
            col_sum.w = select(col_sum.w, floor(col_sum.w / q + 0.5) * q, q > 0.0001);
            // Quantize chroma (Cb, Cr) with stronger quantization
            col_sum.y = select(col_sum.y, floor(col_sum.y / chroma_q + 0.5) * chroma_q, chroma_q > 0.0001);
            col_sum.z = select(col_sum.z, floor(col_sum.z / chroma_q + 0.5) * chroma_q, chroma_q > 0.0001);

            // Zero out high frequency chroma coefficients
            let half_bs = bs / 2u;
            let zero_422 = mode == 1u && lx >= half_bs;
            let zero_420 = mode == 2u && (lx >= half_bs || ly >= half_bs);
            let zero_411 = mode == 3u && lx >= (bs / 4u);
            let zero_chroma = zero_422 || zero_420 || zero_411;
            col_sum.y = select(col_sum.y, 0.0, zero_chroma);
            col_sum.z = select(col_sum.z, 0.0, zero_chroma);

            block[ly][lx] = col_sum;
        }
    }
    workgroupBarrier();

    // Inverse DCT on columns
    for (var ly = lid.y; ly < bs; ly = ly + WORKGROUP_SIZE) {
        for (var lx = lid.x; lx < bs; lx = lx + WORKGROUP_SIZE) {
            var icol_sum = vec4<f32>(0.0);
            for (var k = 0u; k < bs; k = k + 1u) {
                icol_sum = icol_sum + dct_coeff(k) * block[k][lx] * dct_basis(ly, k, bs);
            }
            icol_sum = icol_sum * dct_norm;
            block[ly][lx] = icol_sum;
        }
    }
    workgroupBarrier();

    // Inverse DCT on rows and write output
    for (var ly = lid.y; ly < bs; ly = ly + WORKGROUP_SIZE) {
        for (var lx = lid.x; lx < bs; lx = lx + WORKGROUP_SIZE) {
            var irow_sum = vec4<f32>(0.0);
            for (var k = 0u; k < bs; k = k + 1u) {
                irow_sum = irow_sum + dct_coeff(k) * block[ly][k] * dct_basis(lx, k, bs);
            }
            irow_sum = irow_sum * dct_norm;

            // Shift back from centered, convert YCbCr back to RGB
            var ycbcr = vec3<f32>(irow_sum.x + 0.5, irow_sum.y, irow_sum.z);
            var alpha = irow_sum.w + 0.5;
            var rgb = ycbcr_to_rgb(ycbcr);

            // Apply error effects to RGB
            let px = vec2<i32>(block_origin.x + i32(lx), block_origin.y + i32(ly));
            rgb = apply_error(rgb, px, block_origin);

            var rgba = vec4<f32>(rgb, alpha);

            // Blend with original and write
            let in_bounds = px.x < i32(params.width) && px.y < i32(params.height);
            if in_bounds {
                let original = swizzle_in(textureLoad(input_image, px, 0));
                rgba = mix(rgba, original, params.blend_original);
                textureStore(output_image, px, swizzle_out(clamp(rgba, vec4<f32>(0.0), vec4<f32>(1.0))));
            }
        }
    }
}
