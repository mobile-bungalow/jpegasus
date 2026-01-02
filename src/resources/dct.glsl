
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
layout(set = 1, binding = 0) uniform Params {
    float quality;
};

#pragma input(image, name="input_image")
layout(set = 0, binding = 0) uniform texture2D input_image;

#pragma target(name="output_image", screen)
layout(rgba8, set = 0, binding = 1) uniform writeonly image2D output_image;

#pragma pass(0)
#pragma relay(name="dct", target="dct_target")
layout(rgba16f, set = 0, binding = 2) uniform writeonly image2D dct;
layout(set = 0, binding = 3) uniform texture2D dct_target;

layout(local_size_x = 16, local_size_y = 16) in;

const float PI = 3.14159265359;
const float SQRT_HALF = 0.707106781;

float dct_coeff(int k) {
    return k == 0 ? SQRT_HALF : 1.0;
}

float dct_basis(int n, int k) {
    return cos((2.0 * float(n) + 1.0) * float(k) * PI / 16.0);
}

vec3 quantize(vec3 v, float q) {
    return q > 0.0001 ? floor(v / q + 0.5) * q : v;
}

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(textureSize(input_image, 0));
    if (any(greaterThanEqual(px, size))) return;

    ivec2 block = (px / 8) * 8;
    ivec2 local = px - block;

    float compression = (101.0 - quality) / 100.0;
    float q = compression * compression * compression * 2.0;

    if (pass_index == 0) {
        vec3 sum = vec3(0.0);
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                vec3 pixel = texelFetch(input_image, clamp(block + ivec2(x, y), ivec2(0), size - 1), 0).rgb - 0.5;
                sum += pixel * dct_basis(x, local.x) * dct_basis(y, local.y);
            }
        }
        sum *= 0.25 * dct_coeff(local.x) * dct_coeff(local.y);
        imageStore(dct, px, vec4(quantize(sum, q), 1.0));
    } else {
        vec3 sum = vec3(0.0);
        for (int v = 0; v < 8; v++) {
            for (int u = 0; u < 8; u++) {
                vec3 coeff = texelFetch(dct_target, clamp(block + ivec2(u, v), ivec2(0), size - 1), 0).rgb;
                sum += dct_coeff(u) * dct_coeff(v) * coeff * dct_basis(local.x, u) * dct_basis(local.y, v);
            }
        }
        imageStore(output_image, px, vec4(clamp(sum * 0.25 + 0.5, 0.0, 1.0), 1.0));
    }
}
