//! BMMQ 바이너리 포맷 로더
//!
//! 포맷: magic(b"BMMQ") + version(u16) + n_tensors(u32)
//! Per tensor: name_len(u16) + name + dtype(u8) + ndim(u8) + shape([u32]) + data_len(u64) + data
//!   dtype=0: f32 raw
//!   dtype=1: i8 + row_scales(f32[]) + row_sums(i32[])
//!   dtype=2: packed2bit + gamma(f32) + row_sums(i32[])

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

// ── dtype 코드 ───────────────────────────────────────
const DTYPE_F32: u8 = 0;
const DTYPE_I8: u8 = 1;
const DTYPE_PACKED2BIT: u8 = 2;

// ── 텐서 데이터 enum ────────────────────────────────

pub enum TensorData {
    F32 {
        data: Vec<f32>,
        shape: Vec<usize>,
    },
    I8Quantized {
        data: Vec<i8>,
        row_scales: Vec<f32>,
        row_sums: Vec<i32>,
        rows: usize,
        cols: usize,
    },
    Packed2Bit {
        data: Vec<u8>,
        gamma: f32,
        row_sums: Vec<i32>,
        rows: usize,
        cols: usize,
        packed_stride: usize,
    },
}

// ── 헬퍼: 바이트 읽기 ──────────────────────────────

fn read_u8(r: &mut impl Read) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16(r: &mut impl Read) -> Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

// ── 2-bit 언팩 (C 커널에서 수행 — Rust 측 불필요) ──

// ── BMMQ 파일 파서 ──────────────────────────────────

pub fn load_bmmq<P: AsRef<Path>>(path: P) -> Result<HashMap<String, TensorData>> {
    let file = File::open(path.as_ref())
        .context(format!("BMMQ 파일 열기 실패: {:?}", path.as_ref()))?;
    let mut r = BufReader::new(file);

    // 매직 검증
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"BMMQ" {
        bail!("BMMQ 매직 불일치: {:?}", magic);
    }

    let version = read_u16(&mut r)?;
    if version != 1 {
        bail!("BMMQ 버전 미지원: {}", version);
    }

    let n_tensors = read_u32(&mut r)? as usize;
    let mut tensors = HashMap::with_capacity(n_tensors);

    for _ in 0..n_tensors {
        // 이름
        let name_len = read_u16(&mut r)? as usize;
        let mut name_buf = vec![0u8; name_len];
        r.read_exact(&mut name_buf)?;
        let name = String::from_utf8(name_buf)
            .context("텐서 이름 UTF-8 디코딩 실패")?;

        // dtype, ndim
        let dtype = read_u8(&mut r)?;
        let ndim = read_u8(&mut r)? as usize;

        // shape
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u32(&mut r)? as usize);
        }

        // data_len
        let data_len = read_u64(&mut r)? as usize;

        // 데이터 읽기
        let mut raw = vec![0u8; data_len];
        r.read_exact(&mut raw)?;

        let tensor_data = match dtype {
            DTYPE_F32 => {
                // raw f32 bytes → Vec<f32>
                let n_floats = raw.len() / 4;
                let mut data = vec![0.0f32; n_floats];
                for i in 0..n_floats {
                    let bytes = [raw[i*4], raw[i*4+1], raw[i*4+2], raw[i*4+3]];
                    data[i] = f32::from_le_bytes(bytes);
                }
                TensorData::F32 { data, shape }
            }
            DTYPE_I8 => {
                let rows = shape[0];
                let cols = shape[1];
                let i8_len = rows * cols;
                let scales_offset = i8_len;
                let sums_offset = scales_offset + rows * 4;

                // i8 data
                let data: Vec<i8> = raw[..i8_len].iter().map(|&b| b as i8).collect();

                // row_scales (f32)
                let mut row_scales = vec![0.0f32; rows];
                for i in 0..rows {
                    let off = scales_offset + i * 4;
                    let bytes = [raw[off], raw[off+1], raw[off+2], raw[off+3]];
                    row_scales[i] = f32::from_le_bytes(bytes);
                }

                // row_sums (i32)
                let mut row_sums = vec![0i32; rows];
                for i in 0..rows {
                    let off = sums_offset + i * 4;
                    let bytes = [raw[off], raw[off+1], raw[off+2], raw[off+3]];
                    row_sums[i] = i32::from_le_bytes(bytes);
                }

                TensorData::I8Quantized { data, row_scales, row_sums, rows, cols }
            }
            DTYPE_PACKED2BIT => {
                let rows = shape[0];
                let cols = shape[1];
                let packed_stride = (cols + 3) / 4;
                let packed_len = rows * packed_stride;

                // packed data
                let packed_data = raw[..packed_len].to_vec();

                // gamma (f32)
                let gamma_off = packed_len;
                let gamma = f32::from_le_bytes([
                    raw[gamma_off], raw[gamma_off+1], raw[gamma_off+2], raw[gamma_off+3]
                ]);

                // row_sums (i32)
                let sums_off = gamma_off + 4;
                let mut row_sums = vec![0i32; rows];
                for i in 0..rows {
                    let off = sums_off + i * 4;
                    let bytes = [raw[off], raw[off+1], raw[off+2], raw[off+3]];
                    row_sums[i] = i32::from_le_bytes(bytes);
                }

                TensorData::Packed2Bit {
                    data: packed_data, gamma, row_sums, rows, cols, packed_stride,
                }
            }
            _ => bail!("미지원 dtype: {} (텐서: {})", dtype, name),
        };

        tensors.insert(name, tensor_data);
    }

    Ok(tensors)
}
