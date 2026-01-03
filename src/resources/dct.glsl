#version 450
#pragma stage(compute)
#pragma utility_block(ShaderInputs)
layout(push_constant) uniform ShaderInputs {
    float time;
    float time_delta;
    float frame_rate;
    uint frame_index;
    vec4 mouse;
    vec4 date;
    vec3 resolution;
    uint pass_index;
};

#pragma input(float, name="quality", default=50.0, min=1.0, max=100.0)
#pragma input(int, name="block_size", default=8, min=2, max=64)
#pragma input(float, name="coefficient_threshold", default=0.0, min=0.0, max=1.0)
#pragma input(float, name="error_rate", default=0.0, min=0.0, max=100.0)
#pragma input(float, name="error_brightness_min", default=0.0, min=-1.0, max=1.0)
#pragma input(float, name="error_brightness_max", default=0.0, min=-1.0, max=1.0)
#pragma input(float, name="error_blue_yellow_min", default=0.0, min=-1.0, max=1.0)
#pragma input(float, name="error_blue_yellow_max", default=0.0, min=-1.0, max=1.0)
#pragma input(float, name="error_red_cyan_min", default=0.0, min=-1.0, max=1.0)
#pragma input(float, name="error_red_cyan_max", default=0.0, min=-1.0, max=1.0)
#pragma input(int, name="seed", default=0, min=0, max=10000)
#pragma input(int, name="error_matte_mode", default=0, values=[0,1], labels=["Luminance","RGB Drive YCbCr"])
#pragma input(bool, name="use_error_matte", default=false)
#pragma input(float, name="blend_original", default=0.0, min=0.0, max=1.0)
#pragma input(bool, name="use_luma_quality", default=false)
#pragma input(bool, name="ae_channel_order", default=true)
layout(set = 1, binding = 0) uniform Params {
    float quality;
    int block_size;
    float coefficient_threshold;
    float error_rate;
    float error_brightness_min;
    float error_brightness_max;
    float error_blue_yellow_min;
    float error_blue_yellow_max;
    float error_red_cyan_min;
    float error_red_cyan_max;
    int seed;
    int error_matte_mode;
    int use_error_matte;
    float blend_original;
    int use_luma_quality;
    int ae_channel_order;
};

#pragma input(image, name="input_image")
layout(set = 0, binding = 0) uniform texture2D input_image;

#pragma input(image, name="error_matte")
layout(set = 0, binding = 8) uniform texture2D error_matte;

#pragma input(image, name="luma_quality_matte")
layout(set = 0, binding = 9) uniform texture2D luma_quality_matte;

#pragma target(name="output_image", screen)
layout(rgba8, set = 0, binding = 1) uniform writeonly image2D output_image;

#pragma pass(0)
#pragma relay(name="dct_rows", target="dct_rows_target")
layout(rgba16f, set = 0, binding = 2) uniform writeonly image2D dct_rows;
layout(set = 0, binding = 3) uniform texture2D dct_rows_target;

#pragma pass(1)
#pragma relay(name="dct_cols", target="dct_cols_target")
layout(rgba16f, set = 0, binding = 4) uniform writeonly image2D dct_cols;
layout(set = 0, binding = 5) uniform texture2D dct_cols_target;

#pragma pass(2)
#pragma relay(name="idct_rows", target="idct_rows_target")
layout(rgba16f, set = 0, binding = 6) uniform writeonly image2D idct_rows;
layout(set = 0, binding = 7) uniform texture2D idct_rows_target;

layout(local_size_x = 16, local_size_y = 16) in;

const float PI = 3.14159265359;
const float SQRT_HALF = 0.707106781;

// AE uses ARGB, shader uses RGBA - swizzle based on ae_channel_order
vec4 swizzle_in(vec4 c) {
    return mix(c, c.gbar, float(ae_channel_order));
}

vec4 swizzle_out(vec4 c) {
    return mix(c, c.argb, float(ae_channel_order));
}

float hash(vec2 p, int s) {
    vec3 p3 = fract(vec3(p.xyx + float(s)) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float dct_coeff(int k) {
    return k == 0 ? SQRT_HALF : 1.0;
}

float dct_basis(int n, int k, int N) {
    return cos((2.0 * float(n) + 1.0) * float(k) * PI / (2.0 * float(N)));
}

vec3 quantize(vec3 v, float q) {
    return q > 0.0001 ? floor(v / q + 0.5) * q : v;
}

vec3 rgb_to_ycbcr(vec3 rgb) {
    return vec3(
        0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b,
        -0.169 * rgb.r - 0.331 * rgb.g + 0.500 * rgb.b,
        0.500 * rgb.r - 0.419 * rgb.g - 0.081 * rgb.b
    );
}

vec3 ycbcr_to_rgb(vec3 ycbcr) {
    float y = ycbcr.x, cb = ycbcr.y, cr = ycbcr.z;
    return vec3(
        y + 1.402 * cr,
        y - 0.344 * cb - 0.714 * cr,
        y + 1.772 * cb
    );
}

vec3 sample_error_matte(ivec2 px) {
    vec4 matte = swizzle_in(texelFetch(error_matte, px, 0));
    float lum = 0.299 * matte.r + 0.587 * matte.g + 0.114 * matte.b;
    float use_lum = float(1 - error_matte_mode);
    return mix(matte.rgb, vec3(lum), use_lum);
}

float sample_luma_quality(ivec2 px) {
    vec4 matte = swizzle_in(texelFetch(luma_quality_matte, px, 0));
    return 0.299 * matte.r + 0.587 * matte.g + 0.114 * matte.b;
}

vec3 apply_error(vec3 rgb, ivec2 px, ivec2 block) {
    vec3 score;
    if (use_error_matte == 0) {
        float block_hash = hash(vec2(block), seed);
        if (block_hash * 100.0 >= error_rate) return rgb;
        score = vec3(hash(vec2(px), seed));
    } else {
        score = sample_error_matte(px);
    }

    vec3 ycbcr = rgb_to_ycbcr(rgb);
    vec3 err_min = vec3(error_brightness_min, error_blue_yellow_min, error_red_cyan_min);
    vec3 err_max = vec3(error_brightness_max, error_blue_yellow_max, error_red_cyan_max);
    vec3 rand = vec3(
            hash(vec2(px), seed + 1),
            hash(vec2(px), seed + 2),
            hash(vec2(px), seed + 3)
        );
    vec3 err = mix(err_min, err_max, rand) * score;
    ycbcr *= 1.0 + err * 4.0;
    return ycbcr_to_rgb(ycbcr);
}

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(textureSize(input_image, 0));
    if (any(greaterThanEqual(px, size))) return;

    int N = block_size;
    ivec2 block = (px / N) * N;
    ivec2 local = px - block;
    float norm = sqrt(2.0 / float(N));

    if (pass_index == 0) {
        // Forward DCT on rows
        vec3 sum = vec3(0.0);
        for (int x = 0; x < N; x++) {
            int sx = block.x + x;
            if (sx >= size.x) break;
            vec3 pixel = swizzle_in(texelFetch(input_image, ivec2(sx, px.y), 0)).rgb - 0.5;
            sum += pixel * dct_basis(x, local.x, N);
        }
        sum *= norm * dct_coeff(local.x);
        imageStore(dct_rows, px, vec4(sum, 1.0));
    } else if (pass_index == 1) {
        // Forward DCT on columns + quantize + effects
        vec3 sum = vec3(0.0);
        for (int y = 0; y < N; y++) {
            int sy = block.y + y;
            if (sy >= size.y) break;
            vec3 row_dct = texelFetch(dct_rows_target, ivec2(px.x, sy), 0).rgb;
            sum += row_dct * dct_basis(y, local.y, N);
        }
        sum *= norm * dct_coeff(local.y);

        int freq = local.x + local.y;
        int max_freq = N * 2 - 2;
        int threshold = int(float(max_freq) * (1.0 - coefficient_threshold));
        if (freq > threshold) {
            sum = vec3(0.0);
        }

        sum = apply_error(sum, px, block);

        float matte_val = sample_luma_quality(px);
        float effective_quality = mix(quality, matte_val * 100.0, float(use_luma_quality));
        float compression = (101.0 - effective_quality) / 100.0;
        float q = compression * compression * compression * 2.0;
        imageStore(dct_cols, px, vec4(quantize(sum, q), 1.0));
    } else if (pass_index == 2) {
        // Inverse DCT on columns
        vec3 sum = vec3(0.0);
        for (int v = 0; v < N; v++) {
            int sy = block.y + v;
            if (sy >= size.y) break;
            vec3 coeff = texelFetch(dct_cols_target, ivec2(px.x, sy), 0).rgb;
            sum += dct_coeff(v) * coeff * dct_basis(local.y, v, N);
        }
        sum *= norm;
        imageStore(idct_rows, px, vec4(sum, 1.0));
    } else {
        vec3 sum = vec3(0.0);
        for (int u = 0; u < N; u++) {
            int sx = block.x + u;
            if (sx >= size.x) break;
            vec3 coeff = texelFetch(idct_rows_target, ivec2(sx, px.y), 0).rgb;
            sum += dct_coeff(u) * coeff * dct_basis(local.x, u, N);
        }
        sum *= norm;

        vec3 rgb = sum + 0.5;
        vec3 original = swizzle_in(texelFetch(input_image, px, 0)).rgb;
        rgb = mix(rgb, original, blend_original);
        imageStore(output_image, px, swizzle_out(vec4(clamp(rgb, 0.0, 1.0), 1.0)));
    }
}
