mod types {
    include!("../src/types.rs");
}
mod pipeline {
    include!("../src/pipeline/mod.rs");
}

use pipeline::{DctPipeline, DctPushConstants};
use std::time::Instant;
use types::BitDepth;

fn gpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let mut limits = wgpu::Limits::default();
    limits.max_push_constant_size = 256;
    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: wgpu::Features::PUSH_CONSTANTS,
            required_limits: limits,
            ..Default::default()
        },
        None,
    ))
    .unwrap()
}

#[test]
fn test_dct_roundtrip() {
    let (device, queue) = gpu();
    let mut pipeline = DctPipeline::new(&device);

    let img = image::open("tests/fixtures/input.png").unwrap().to_rgba8();
    let (w, h) = img.dimensions();
    let input: Vec<u8> = img.into_raw();
    let mut output = vec![0u8; input.len()];

    let row_bytes = (w * 4) as usize;
    let mut params = DctPushConstants {
        block_size: 8,
        ..DctPushConstants::new()
    };
    params.set_quality(50.0);
    pipeline.render(
        &device,
        &queue,
        params,
        &input,
        &mut output,
        w,
        h,
        row_bytes,
        row_bytes,
        BitDepth::U8,
        None,
        None,
    );

    let out_img = image::RgbaImage::from_raw(w, h, output).unwrap();
    out_img.save("tests/fixtures/output.png").unwrap();
}

#[test]
fn bench_render() {
    let (device, queue) = gpu();
    let mut pipeline = DctPipeline::new(&device);

    let w = 1920u32;
    let h = 1080u32;
    let input: Vec<u8> = (0..(w * h * 4)).map(|i| (i % 256) as u8).collect();
    let mut output = vec![0u8; input.len()];
    let row_bytes = (w * 4) as usize;

    println!("\n=== Benchmark Results (1920x1080) ===");

    for block_size in [8u32, 16, 32, 64] {
        let mut params = DctPushConstants {
            block_size,
            ..DctPushConstants::new()
        };
        params.set_quality(50.0);

        // Warmup
        pipeline.render(
            &device,
            &queue,
            params,
            &input,
            &mut output,
            w,
            h,
            row_bytes,
            row_bytes,
            BitDepth::U8,
            None,
            None,
        );

        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            pipeline.render(
                &device,
                &queue,
                params,
                &input,
                &mut output,
                w,
                h,
                row_bytes,
                row_bytes,
                BitDepth::U8,
                None,
                None,
            );
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_millis() as f64 / iterations as f64;
        let fps = 1000.0 / avg_ms;

        println!("Block {block_size}x{block_size}: {avg_ms:.1}ms ({fps:.1} FPS)");
    }

    println!("=====================================\n");
}

#[test]
fn bench_workgroup_sizes() {
    // Note: This test documents the workgroup size we're using.
    // wgpu guarantees at least 256 threads per workgroup (16x16 or 8x32).
    // Our 8x8 = 64 threads is well within limits and matches JPEG block size.
    // Common guaranteed limits:
    // - max_compute_workgroup_size_x: 256
    // - max_compute_workgroup_size_y: 256
    // - max_compute_workgroups_per_dimension: 65535
    // - max_compute_invocations_per_workgroup: 256

    println!("\n=== Workgroup Size Analysis ===");
    println!("Current: 8x8 = 64 threads (matches JPEG block)");
    println!("wgpu guarantees: max 256 invocations per workgroup");
    println!("Safe sizes: 8x8, 16x16, 8x32, 32x8");
    println!("================================\n");
}

fn create_test_image(w: u32, h: u32) -> (Vec<u8>, image::RgbaImage) {
    let mut img = image::RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            // Create a colorful gradient with distinct chroma information
            let r = ((x * 255) / w) as u8;
            let g = ((y * 255) / h) as u8;
            let b = (((x + y) * 127) / (w + h)) as u8;
            img.put_pixel(x, y, image::Rgba([r, g, b, 255]));
        }
    }
    let raw = img.clone().into_raw();
    (raw, img)
}

fn image_similarity(a: &[u8], b: &[u8], w: u32, h: u32) -> f64 {
    let img_a = image::RgbaImage::from_raw(w, h, a.to_vec()).unwrap();
    let img_b = image::RgbaImage::from_raw(w, h, b.to_vec()).unwrap();

    let gray_a = image::DynamicImage::ImageRgba8(img_a).into_luma8();
    let gray_b = image::DynamicImage::ImageRgba8(img_b).into_luma8();

    let result = image_compare::gray_similarity_structure(
        &image_compare::Algorithm::RootMeanSquared,
        &gray_a,
        &gray_b,
    )
    .unwrap();

    result.score
}

#[test]
fn test_chroma_subsampling_modes_differ() {
    let (device, queue) = gpu();
    let mut pipeline = DctPipeline::new(&device);

    let w = 128u32;
    let h = 128u32;
    let (input, _) = create_test_image(w, h);
    let row_bytes = (w * 4) as usize;

    // Chroma subsampling modes: 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0, 3 = 4:1:1
    let mode_names = ["4:4:4 (None)", "4:2:2", "4:2:0", "4:1:1"];
    let mut outputs: Vec<Vec<u8>> = Vec::new();

    for mode in 0u32..4 {
        let mut output = vec![0u8; input.len()];
        let mut params = DctPushConstants {
            block_size: 16,
            chroma_subsampling: mode,
            ..DctPushConstants::new()
        };
        params.set_quality(50.0);

        println!(
            "Rendering with chroma_subsampling={}, quantization_step={}",
            params.chroma_subsampling, params.quantization_step
        );

        pipeline.render(
            &device,
            &queue,
            params,
            &input,
            &mut output,
            w,
            h,
            row_bytes,
            row_bytes,
            BitDepth::U8,
            None,
            None,
        );

        outputs.push(output);
    }

    // Save outputs for visual inspection
    for (i, output) in outputs.iter().enumerate() {
        let img = image::RgbaImage::from_raw(w, h, output.clone()).unwrap();
        img.save(format!("tests/fixtures/chroma_{}.png", i))
            .unwrap();
    }

    // Also save input for reference
    let input_img = image::RgbaImage::from_raw(w, h, input.clone()).unwrap();
    input_img.save("tests/fixtures/chroma_input.png").unwrap();

    // Compare each pair of modes using image similarity
    println!("\n=== Chroma Subsampling Mode Comparison ===");
    for i in 0..4 {
        for j in (i + 1)..4 {
            let similarity = image_similarity(&outputs[i], &outputs[j], w, h);
            println!(
                "{} vs {}: similarity = {:.4}",
                mode_names[i], mode_names[j], similarity
            );

            // Images should be similar but not identical (similarity < 1.0)
            assert!(
                similarity < 0.9999,
                "Expected {} and {} to produce different outputs (similarity={:.4})",
                mode_names[i],
                mode_names[j],
                similarity
            );
        }
    }
    println!("==========================================\n");
}

#[test]
fn test_block_sizes_differ() {
    let (device, queue) = gpu();
    let mut pipeline = DctPipeline::new(&device);

    let w = 128u32;
    let h = 128u32;
    let (input, _) = create_test_image(w, h);
    let row_bytes = (w * 4) as usize;

    // Test different block sizes (effect visible for sizes <= 16)
    let block_sizes = [8u32, 10, 12, 16];
    let mut outputs: Vec<Vec<u8>> = Vec::new();

    for &block_size in &block_sizes {
        let mut output = vec![0u8; input.len()];
        let mut params = DctPushConstants {
            block_size,
            ..DctPushConstants::new()
        };
        params.set_quality(50.0);

        pipeline.render(
            &device,
            &queue,
            params,
            &input,
            &mut output,
            w,
            h,
            row_bytes,
            row_bytes,
            BitDepth::U8,
            None,
            None,
        );

        outputs.push(output);
    }

    // Compare each pair of block sizes
    println!("\n=== Block Size Comparison ===");
    for i in 0..block_sizes.len() {
        for j in (i + 1)..block_sizes.len() {
            let similarity = image_similarity(&outputs[i], &outputs[j], w, h);
            println!(
                "Block {}x{} vs {}x{}: similarity = {:.4}",
                block_sizes[i], block_sizes[i], block_sizes[j], block_sizes[j], similarity
            );

            assert!(
                similarity < 0.9999,
                "Expected block size {} and {} to produce different outputs (similarity={:.4})",
                block_sizes[i],
                block_sizes[j],
                similarity
            );
        }
    }
    println!("=============================\n");
}
