struct Params {
    width: u32,
    height: u32,
    block_size: u32,
    quantization_step: f32,
    coefficient_min: f32,
    coefficient_max: f32,
    blend_original: f32,
    use_luma_quality: u32,
    ae_channel_order: u32,
    use_ycbcr: u32,  // 0 = RGB, 1 = YCbCr
}

var<immediate> params: Params;

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var dct_read: texture_2d<f32>;
@group(0) @binding(2) var dct_write: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var quality_tex: texture_2d<f32>;
@group(0) @binding(4) var output_tex: texture_storage_2d<rgba8unorm, write>;

const PI: f32 = 3.14159265359;
const SQRT_HALF: f32 = 0.70710678118;

// BT.601 color space
const KR: f32 = 0.299;
const KG: f32 = 0.587;
const KB: f32 = 0.114;
const CB_R: f32 = -0.168736;
const CB_G: f32 = -0.331264;
const CR_G: f32 = -0.418688;
const CR_B: f32 = -0.081312;
const R_CR: f32 = 1.402;
const G_CB: f32 = -0.344136;
const G_CR: f32 = -0.714136;
const B_CB: f32 = 1.772;

fn in_bounds(px: vec2<i32>) -> bool {
    return px.x >= 0 && px.x < i32(params.width) && px.y >= 0 && px.y < i32(params.height);
}

fn block_local(px: vec2<u32>, bs: u32) -> vec2<u32> { return px % bs; }
fn block_origin(px: vec2<u32>, bs: u32) -> vec2<i32> { return vec2<i32>((px / bs) * bs); }

fn load_pixel(tex: texture_2d<f32>, px: vec2<i32>) -> vec4<f32> {
    var c = textureLoad(tex, px, 0);
    if params.ae_channel_order == 1u { c = c.gbar; }
    return c;
}

fn store_output(px: vec2<i32>, c: vec4<f32>) {
    var out = c;
    if params.ae_channel_order == 1u { out = out.argb; }
    textureStore(output_tex, px, out);
}

fn rgb_to_ycbcr(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        KR * c.r + KG * c.g + KB * c.b - 0.5,
        CB_R * c.r + CB_G * c.g + 0.5 * c.b,
        0.5 * c.r + CR_G * c.g + CR_B * c.b,
        c.a - 0.5
    );
}

fn ycbcr_to_rgb(c: vec4<f32>) -> vec4<f32> {
    let y = c.x + 0.5;
    let cb = c.y;
    let cr = c.z;
    let a = c.w + 0.5;
    return vec4<f32>(y + R_CR * cr, y + G_CB * cb + G_CR * cr, y + B_CB * cb, a);
}

fn to_working_space(c: vec4<f32>) -> vec4<f32> {
    if params.use_ycbcr == 1u {
        return rgb_to_ycbcr(c);
    }
    return c - 0.5;
}

fn from_working_space(c: vec4<f32>) -> vec4<f32> {
    if params.use_ycbcr == 1u {
        return ycbcr_to_rgb(c);
    }
    return c + 0.5;
}

fn dct_basis(n: u32, k: u32, N: f32) -> f32 { return cos((2.0 * f32(n) + 1.0) * f32(k) * PI / (2.0 * N)); }
fn dct_scale(k: u32) -> f32 { return select(1.0, SQRT_HALF, k == 0u); }
fn dct_norm(N: f32) -> f32 { return sqrt(2.0 / N); }

fn quantize(v: f32, step: f32) -> f32 { return floor(v / step + 0.5) * step; }

@compute @workgroup_size(8, 8)
fn pass_dct_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = vec2<i32>(gid.xy);
    if !in_bounds(px) { return; }

    let bs = params.block_size;
    let bsf = f32(bs);
    let k = block_local(vec2<u32>(gid.xy), bs).x;
    let origin = block_origin(vec2<u32>(gid.xy), bs);

    var sum = vec4<f32>(0.0);
    for (var n = 0u; n < bs; n++) {
        sum += to_working_space(load_pixel(input_tex, vec2<i32>(origin.x + i32(n), px.y))) * dct_basis(n, k, bsf);
    }
    textureStore(dct_write, px, sum * dct_norm(bsf) * dct_scale(k));
}

@compute @workgroup_size(8, 8)
fn pass_dct_cols(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = vec2<i32>(gid.xy);
    if !in_bounds(px) { return; }

    let bs = params.block_size;
    let bsf = f32(bs);
    let k = block_local(vec2<u32>(gid.xy), bs).y;
    let origin = block_origin(vec2<u32>(gid.xy), bs);

    var sum = vec4<f32>(0.0);
    for (var n = 0u; n < bs; n++) {
        sum += textureLoad(dct_read, vec2<i32>(px.x, origin.y + i32(n)), 0) * dct_basis(n, k, bsf);
    }
    textureStore(dct_write, px, sum * dct_norm(bsf) * dct_scale(k));
}

@compute @workgroup_size(8, 8)
fn pass_quantize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = vec2<i32>(gid.xy);
    if !in_bounds(px) { return; }

    let bs = params.block_size;
    let local = block_local(vec2<u32>(gid.xy), bs);
    var coef = textureLoad(dct_read, px, 0);

    let freq = local.x + local.y;
    let max_freq = bs * 2u - 2u;
    if freq < u32(f32(max_freq) * params.coefficient_min) || freq > u32(f32(max_freq) * params.coefficient_max) {
        textureStore(dct_write, px, vec4<f32>(0.0));
        return;
    }

    var q = params.quantization_step;
    if params.use_luma_quality == 1u {
        let m = load_pixel(quality_tex, px);
        let compression = (101.0 - (KR * m.r + KG * m.g + KB * m.b) * 100.0) / 100.0;
        q = compression * compression * compression * 2.0;
    }
    if q > 0.0001 {
        coef = vec4<f32>(quantize(coef.x, q), quantize(coef.y, q), quantize(coef.z, q), quantize(coef.w, q));
    }
    textureStore(dct_write, px, coef);
}

@compute @workgroup_size(8, 8)
fn pass_idct_cols(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = vec2<i32>(gid.xy);
    if !in_bounds(px) { return; }

    let bs = params.block_size;
    let bsf = f32(bs);
    let n = block_local(vec2<u32>(gid.xy), bs).y;
    let origin = block_origin(vec2<u32>(gid.xy), bs);

    var sum = vec4<f32>(0.0);
    for (var k = 0u; k < bs; k++) {
        sum += dct_scale(k) * textureLoad(dct_read, vec2<i32>(px.x, origin.y + i32(k)), 0) * dct_basis(n, k, bsf);
    }
    textureStore(dct_write, px, sum * dct_norm(bsf));
}

@compute @workgroup_size(8, 8)
fn pass_idct_rows_final(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = vec2<i32>(gid.xy);
    if !in_bounds(px) { return; }

    let bs = params.block_size;
    let bsf = f32(bs);
    let n = block_local(vec2<u32>(px), bs).x;
    let origin = block_origin(vec2<u32>(px), bs);

    var sum = vec4<f32>(0.0);
    for (var k = 0u; k < bs; k++) {
        sum += dct_scale(k) * textureLoad(dct_read, vec2<i32>(origin.x + i32(k), px.y), 0) * dct_basis(n, k, bsf);
    }
    let result = from_working_space(sum * dct_norm(bsf));
    let original = load_pixel(input_tex, px);
    var final_result = clamp(mix(result, original, params.blend_original), vec4<f32>(0.0), vec4<f32>(1.0));

    // Preserve fully opaque alpha
    if original.a >= 1.0 {
        final_result.a = 1.0;
    }

    store_output(px, final_result);
}
