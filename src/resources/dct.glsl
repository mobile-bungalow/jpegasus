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
#pragma input(int, name="coeff_kill", default=0, min=0, max=64)
#pragma input(float, name="error_rate", default=0.0, min=0.0, max=100.0)
#pragma input(float, name="error_severity", default=0.5, min=0.0, max=1.0)

#pragma input(int, name="seed", default=0, min=0, max=10000)
layout(set = 1, binding = 0) uniform Params {
    float quality;
    int block_size;
    int coeff_kill;
    float error_rate;
    float error_severity;
    int seed;
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

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(textureSize(input_image, 0));
    if (any(greaterThanEqual(px, size))) return;

    int N = block_size;
    ivec2 block = (px / N) * N;
    ivec2 local = px - block;

    float compression = (101.0 - quality) / 100.0;
    float q = compression * compression * compression * 2.0;

    if (pass_index == 0) {
        vec3 sum = vec3(0.0);
        for (int y = 0; y < N; y++) {
            int sy = block.y + y;
            if (sy >= size.y) break;
            for (int x = 0; x < N; x++) {
                int sx = block.x + x;
                if (sx >= size.x) break;
                vec3 pixel = texelFetch(input_image, ivec2(sx, sy), 0).rgb - 0.5;
                sum += pixel * dct_basis(x, local.x, N) * dct_basis(y, local.y, N);
            }
        }
        sum *= (2.0 / float(N)) * dct_coeff(local.x) * dct_coeff(local.y);

        int freq = local.x + local.y;
        if (freq >= N * 2 - coeff_kill - 1) {
            sum = vec3(0.0);
        }

        float block_hash = hash(vec2(block), seed);
        if (block_hash * 100.0 < error_rate) {
            float err = (hash(vec2(px), seed + 1) * 2.0 - 1.0) * error_severity;
            sum = mix(sum, sum * (1.0 + err * 4.0), error_severity);
        }

        imageStore(dct, px, vec4(quantize(sum, q), 1.0));
    } else {
        vec3 sum = vec3(0.0);
        for (int v = 0; v < N; v++) {
            if (block.y + v >= size.y) break;
            for (int u = 0; u < N; u++) {
                if (block.x + u >= size.x) break;
                vec3 coeff = texelFetch(dct_target, min(block + ivec2(u, v), size - 1), 0).rgb;
                sum += dct_coeff(u) * dct_coeff(v) * coeff * dct_basis(local.x, u, N) * dct_basis(local.y, v, N);
            }
        }

        vec3 rgb = sum * (2.0 / float(N)) + 0.5;
        imageStore(output_image, px, vec4(clamp(rgb, 0.0, 1.0), 1.0));
    }
}
