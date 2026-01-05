use crate::types::BitDepth;
use std::mem::size_of;

const CHANNELS: usize = 4;
const PIXEL_SIZE: usize = CHANNELS * size_of::<u8>();

pub fn load_rbga8_tex(
    dst: &mut Vec<u8>,
    src: &[u8],
    depth: BitDepth,
    w: u32,
    h: u32,
    src_stride: usize,
) {
    let w = w as usize;
    let h = h as usize;
    let dst_stride = w * PIXEL_SIZE;
    dst.resize(h * dst_stride, 0);

    match depth {
        BitDepth::U8 | BitDepth::Invalid(_) => {
            for (src_row, dst_row) in src
                .chunks(src_stride)
                .zip(dst.chunks_mut(dst_stride))
                .take(h)
            {
                dst_row.copy_from_slice(&src_row[..dst_stride]);
            }
        }
        BitDepth::U16 => {
            let src_bpp = CHANNELS * size_of::<u16>();
            for (src_row, dst_row) in src
                .chunks(src_stride)
                .zip(dst.chunks_mut(dst_stride))
                .take(h)
            {
                for (src_px, dst_px) in src_row
                    .chunks_exact(src_bpp)
                    .zip(dst_row.chunks_exact_mut(PIXEL_SIZE))
                {
                    for c in 0..CHANNELS {
                        let offset = c * size_of::<u16>();
                        let bytes = &src_px[offset..offset + size_of::<u16>()];
                        dst_px[c] = bytes[1];
                    }
                }
            }
        }
        BitDepth::F32 => {
            let src_bpp = CHANNELS * size_of::<f32>();
            for (src_row, dst_row) in src
                .chunks(src_stride)
                .zip(dst.chunks_mut(dst_stride))
                .take(h)
            {
                for (src_px, dst_px) in src_row
                    .chunks_exact(src_bpp)
                    .zip(dst_row.chunks_exact_mut(PIXEL_SIZE))
                {
                    for c in 0..CHANNELS {
                        let offset = c * size_of::<f32>();
                        let bytes = &src_px[offset..offset + size_of::<f32>()];
                        let f = f32::from_ne_bytes(bytes.try_into().unwrap());
                        dst_px[c] = (f.clamp(0.0, 1.0) * 255.0) as u8;
                    }
                }
            }
        }
    }
}

pub fn copy_rows(src: &[u8], dst: &mut [u8], w: u32, h: u32, src_stride: usize, dst_stride: usize) {
    let row_len = w as usize * CHANNELS * size_of::<u8>();
    for (src_row, dst_row) in src
        .chunks(src_stride)
        .zip(dst.chunks_mut(dst_stride))
        .take(h as usize)
    {
        dst_row[..row_len].copy_from_slice(&src_row[..row_len]);
    }
}

pub fn from_8bit_to_16(
    src: &[u8],
    dst: &mut [u8],
    w: u32,
    h: u32,
    src_stride: usize,
    dst_stride: usize,
) {
    let dst_bpp = CHANNELS * size_of::<u16>();
    for (src_row, dst_row) in src
        .chunks(src_stride)
        .zip(dst.chunks_mut(dst_stride))
        .take(h as usize)
    {
        for (src_px, dst_px) in src_row[..w as usize * PIXEL_SIZE]
            .chunks_exact(PIXEL_SIZE)
            .zip(dst_row.chunks_exact_mut(dst_bpp))
        {
            for (src_byte, dst_bytes) in
                src_px.iter().zip(dst_px.chunks_exact_mut(size_of::<u16>()))
            {
                let v16 = u16::from_ne_bytes([*src_byte, *src_byte]);
                dst_bytes.copy_from_slice(&v16.to_ne_bytes());
            }
        }
    }
}

/// Convert 8-bit GPU output to 32-bit float AE format.
pub fn from_8bit_to_f32(
    src: &[u8],
    dst: &mut [u8],
    w: u32,
    h: u32,
    src_stride: usize,
    dst_stride: usize,
) {
    let dst_bpp = CHANNELS * size_of::<f32>();
    for (src_row, dst_row) in src
        .chunks(src_stride)
        .zip(dst.chunks_mut(dst_stride))
        .take(h as usize)
    {
        for (src_px, dst_px) in src_row[..w as usize * PIXEL_SIZE]
            .chunks_exact(PIXEL_SIZE)
            .zip(dst_row.chunks_exact_mut(dst_bpp))
        {
            for (src_byte, dst_bytes) in
                src_px.iter().zip(dst_px.chunks_exact_mut(size_of::<f32>()))
            {
                let f = *src_byte as f32 / 255.0;
                dst_bytes.copy_from_slice(&f.to_ne_bytes());
            }
        }
    }
}
