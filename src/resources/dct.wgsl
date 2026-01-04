const PI: f32 = 3.14159265359;
const SQRT_HALF: f32 = 0.707106781;

struct Params {
    width: u32,
    height: u32,
    pass_index: u32,
    block_size: u32,
    quantization_step: f32,
    coefficient_min: f32,
    coefficient_max: f32,
    blend_original: f32,
    use_luma_quality: u32,
    ae_channel_order: u32,
    chroma_subsampling: u32,
}

var<push_constant> params: Params;

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var coef_tex: texture_2d<f32>;
@group(0) @binding(3) var _unused_tex: texture_2d<f32>;
@group(0) @binding(4) var quality_matte_tex: texture_2d<f32>;
@group(0) @binding(5) var output_tex_8bit: texture_storage_2d<rgba8unorm, write>;

// DCT basis function: cos((2n+1)k*pi/2N)
fn dct_basis(n: u32, k: u32, block_size: f32) -> f32 {
    return cos((2.0 * f32(n) + 1.0) * f32(k) * PI / (2.0 * block_size));
}

// DC coefficient scaling factor
fn dc_scale(k: u32) -> f32 {
    return select(1.0, SQRT_HALF, k == 0u);
}

// RGB to YCbCr (centered at 0 for DCT)
fn to_ycbcra(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
         0.299 * c.r + 0.587 * c.g + 0.114 * c.b - 0.5,
        -0.169 * c.r - 0.331 * c.g + 0.500 * c.b,
         0.500 * c.r - 0.419 * c.g - 0.081 * c.b,
        c.a - 0.5
    );
}

// YCbCr to RGB
fn to_rgba(y: f32, cb: f32, cr: f32, a: f32) -> vec4<f32> {
    return vec4<f32>(
        y + 1.402 * cr,
        y - 0.344 * cb - 0.714 * cr,
        y + 1.772 * cb,
        a
    );
}

// Quantize a value to nearest multiple of step
fn quantize(v: f32, step: f32) -> f32 {
    return floor(v / step + 0.5) * step;
}

// Load pixel with AE channel order correction
fn load_pixel(tex: texture_2d<f32>, px: vec2<i32>) -> vec4<f32> {
    var c = textureLoad(tex, px, 0);
    if params.ae_channel_order == 1u { c = c.gbar; }
    return c;
}

// Store pixel with AE channel order correction (16-bit float)
fn store_pixel(px: vec2<i32>, c: vec4<f32>) {
    var out = c;
    if params.ae_channel_order == 1u { out = out.argb; }
    textureStore(output_tex, px, out);
}

// Store pixel with AE channel order correction (8-bit)
fn store_pixel_8bit(px: vec2<i32>, c: vec4<f32>) {
    var out = c;
    if params.ae_channel_order == 1u { out = out.argb; }
    textureStore(output_tex_8bit, px, out);
}

// Get block-local coordinates
fn block_local(px: vec2<u32>, bs: u32) -> vec2<u32> {
    return vec2<u32>(px.x % bs, px.y % bs);
}

fn block_origin(px: vec2<u32>, bs: u32) -> vec2<i32> {
    return vec2<i32>(i32((px.x / bs) * bs), i32((px.y / bs) * bs));
}

// Pass 0: RGB to YCbCr
@compute @workgroup_size(8, 8)
fn pass_rgb_to_ycbcr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = vec2<i32>(i32(gid.x), i32(gid.y));
    if px.x >= i32(params.width) || px.y >= i32(params.height) { return; }

    let rgb = load_pixel(input_tex, px);
    let ycbcr = to_ycbcra(rgb);
    textureStore(output_tex, px, ycbcr);
}

// Pass 1: Forward DCT on rows
@compute @workgroup_size(8, 8)
fn pass_dct_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = vec2<i32>(i32(gid.x), i32(gid.y));
    if px.x >= i32(params.width) || px.y >= i32(params.height) { return; }

    let bs = params.block_size;
    let bsf = f32(bs);
    let norm = sqrt(2.0 / bsf);
    let local = block_local(vec2<u32>(gid.xy), bs);
    let origin = block_origin(vec2<u32>(gid.xy), bs);
    let k = local.x;

    var sum = vec4<f32>(0.0);
    for (var n = 0u; n < bs; n++) {
        let sample_px = origin + vec2<i32>(i32(n), 0);
        let v = textureLoad(input_tex, vec2<i32>(sample_px.x, px.y), 0);
        sum += v * dct_basis(n, k, bsf);
    }

    let coef = sum * norm * dc_scale(k);
    textureStore(output_tex, px, coef);
}

// Pass 2: Forward DCT on columns
@compute @workgroup_size(8, 8)
fn pass_dct_cols(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = vec2<i32>(i32(gid.x), i32(gid.y));
    if px.x >= i32(params.width) || px.y >= i32(params.height) { return; }

    let bs = params.block_size;
    let bsf = f32(bs);
    let norm = sqrt(2.0 / bsf);
    let local = block_local(vec2<u32>(gid.xy), bs);
    let origin = block_origin(vec2<u32>(gid.xy), bs);
    let k = local.y;

    var sum = vec4<f32>(0.0);
    for (var n = 0u; n < bs; n++) {
        let sample_px = origin + vec2<i32>(0, i32(n));
        let v = textureLoad(input_tex, vec2<i32>(px.x, sample_px.y), 0);
        sum += v * dct_basis(n, k, bsf);
    }

    let coef = sum * norm * dc_scale(k);
    textureStore(output_tex, px, coef);
}

// Pass 3: Quantize coefficients with error injection
@compute @workgroup_size(8, 8)
fn pass_quantize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = vec2<i32>(i32(gid.x), i32(gid.y));
    if px.x >= i32(params.width) || px.y >= i32(params.height) { return; }

    let bs = params.block_size;
    let local = block_local(vec2<u32>(gid.xy), bs);
    let origin = block_origin(vec2<u32>(gid.xy), bs);
    var coef = textureLoad(input_tex, px, 0);

    // Coefficient range: zero frequencies outside [min, max]
    let freq = local.x + local.y;
    let max_freq = bs * 2u - 2u;
    let min_freq = u32(f32(max_freq) * params.coefficient_min);
    let max_allowed = u32(f32(max_freq) * params.coefficient_max);
    if freq < min_freq || freq > max_allowed {
        textureStore(output_tex, px, vec4<f32>(0.0));
        return;
    }

    // Get quantization step (possibly from quality matte)
    var q = params.quantization_step;
    if params.use_luma_quality == 1u {
        let m = load_pixel(quality_matte_tex, px);
        let lum = 0.299 * m.r + 0.587 * m.g + 0.114 * m.b;
        let compression = (101.0 - lum * 100.0) / 100.0;
        q = compression * compression * compression * 2.0;
    }

    // Chroma gets stronger quantization based on subsampling mode
    let chroma_mult = select(1.0,
        select(2.0, select(4.0, 8.0, params.chroma_subsampling == 3u),
        params.chroma_subsampling >= 2u),
        params.chroma_subsampling >= 1u);

    if q > 0.0001 {
        coef.x = quantize(coef.x, q);           // Y
        coef.y = quantize(coef.y, q * chroma_mult); // Cb
        coef.z = quantize(coef.z, q * chroma_mult); // Cr
        coef.w = quantize(coef.w, q);           // A
    }

    // Chroma subsampling: zero high-frequency chroma
    let half = bs / 2u;
    let quarter = bs / 4u;
    let zero_chroma =
        (params.chroma_subsampling == 1u && local.x >= half) ||
        (params.chroma_subsampling == 2u && (local.x >= half || local.y >= half)) ||
        (params.chroma_subsampling == 3u && local.x >= quarter);
    if zero_chroma {
        coef.y = 0.0;
        coef.z = 0.0;
    }

    textureStore(output_tex, px, coef);
}

// Pass 4: Inverse DCT on columns
@compute @workgroup_size(8, 8)
fn pass_idct_cols(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = vec2<i32>(i32(gid.x), i32(gid.y));
    if px.x >= i32(params.width) || px.y >= i32(params.height) { return; }

    let bs = params.block_size;
    let bsf = f32(bs);
    let norm = sqrt(2.0 / bsf);
    let local = block_local(vec2<u32>(gid.xy), bs);
    let origin = block_origin(vec2<u32>(gid.xy), bs);
    let n = local.y;

    var sum = vec4<f32>(0.0);
    for (var k = 0u; k < bs; k++) {
        let sample_px = origin + vec2<i32>(0, i32(k));
        let coef = textureLoad(input_tex, vec2<i32>(px.x, sample_px.y), 0);
        sum += dc_scale(k) * coef * dct_basis(n, k, bsf);
    }

    textureStore(output_tex, px, sum * norm);
}

// Pass 5: Inverse DCT on rows
@compute @workgroup_size(8, 8)
fn pass_idct_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = vec2<i32>(i32(gid.x), i32(gid.y));
    if px.x >= i32(params.width) || px.y >= i32(params.height) { return; }

    let bs = params.block_size;
    let bsf = f32(bs);
    let norm = sqrt(2.0 / bsf);
    let local = block_local(vec2<u32>(gid.xy), bs);
    let origin = block_origin(vec2<u32>(gid.xy), bs);
    let n = local.x;

    var sum = vec4<f32>(0.0);
    for (var k = 0u; k < bs; k++) {
        let sample_px = origin + vec2<i32>(i32(k), 0);
        let coef = textureLoad(input_tex, vec2<i32>(sample_px.x, px.y), 0);
        sum += dc_scale(k) * coef * dct_basis(n, k, bsf);
    }

    textureStore(output_tex, px, sum * norm);
}

// Finalize: YCbCr to RGB with blending
fn finalize(px: vec2<i32>) -> vec4<f32> {
    let ycbcr = textureLoad(input_tex, px, 0);
    let rgb = to_rgba(ycbcr.x + 0.5, ycbcr.y, ycbcr.z, ycbcr.w + 0.5);
    let orig = load_pixel(coef_tex, px);
    return clamp(mix(rgb, orig, params.blend_original), vec4<f32>(0.0), vec4<f32>(1.0));
}

@compute @workgroup_size(8, 8)
fn pass_finalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = vec2<i32>(i32(gid.x), i32(gid.y));
    if px.x >= i32(params.width) || px.y >= i32(params.height) { return; }
    store_pixel(px, finalize(px));
}

@compute @workgroup_size(8, 8)
fn pass_finalize_8bit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = vec2<i32>(i32(gid.x), i32(gid.y));
    if px.x >= i32(params.width) || px.y >= i32(params.height) { return; }
    store_pixel_8bit(px, finalize(px));
}
