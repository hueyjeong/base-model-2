//! 전체 모델 엔드-투-엔드 벤치마크 v2 — 최적화 + mLSTM
//!
//! 최적화:
//! 1. Fused projection: N개 d→d를 1회 d→Nd sgemm + 양자화 1회 공유
//! 2. FFN gate+up fused (양자화 1회)
//! 3. Vectorized scan: fast_exp/sigmoid/tanh AVX2
//! 4. mLSTM 추가 (matrix memory O(d²))
//!
//! 실제 n_layers 전부 실행

use std::time::Instant;
use crate::common::*;

#[allow(non_camel_case_types)]
type c_int = i32;

fn make_dummy_i8(rows: usize, cols: usize) -> (Vec<i8>, Vec<i32>) {
    let mut w = vec![0i8; rows * cols];
    let mut rs = vec![0i32; rows];
    for r in 0..rows {
        let mut s = 0i32;
        for c in 0..cols {
            let v = match (r*7+c*13)%5 { 0=>1i8, 1=>-1i8, _=>0i8 };
            w[r*cols+c] = v; s += v as i32;
        }
        rs[r] = s;
    }
    (w, rs)
}

fn rms_norm_batch_affine(x: &[f32], w: &[f32], eps: f32, sl: usize, d: usize, out: &mut [f32]) {
    for t in 0..sl {
        let b = t*d;
        let mut sq = 0.0f32;
        for i in 0..d { sq += x[b+i]*x[b+i]; }
        let rms = (sq/d as f32 + eps).sqrt().recip();
        for i in 0..d { out[b+i] = x[b+i]*rms*w[i]; }
    }
}

/// 양자화 1회 + sgemm 1회 (fused)
fn quant_sgemm(
    x: &[f32], sl: usize, in_d: usize, out_d: usize,
    w: &[i8], rs: &[i32], gamma: f32,
    out: &mut [f32],
    nb: &mut Vec<f32>, ub: &mut Vec<u8>, sb: &mut Vec<f32>,
) {
    nb.resize(sl*in_d, 0.0);
    for t in 0..sl { rms_norm_no_affine(&x[t*in_d..(t+1)*in_d], &mut nb[t*in_d..(t+1)*in_d], 1e-5); }
    ub.resize(sl*in_d, 0); sb.resize(sl, 0.0);
    unsafe {
        batch_quantize_f32_to_u8(nb.as_ptr(), ub.as_mut_ptr(), sb.as_mut_ptr(), sl as c_int, in_d as c_int);
        i8_sgemm(w.as_ptr(), ub.as_ptr(), out.as_mut_ptr(),
            out_d as c_int, sl as c_int, in_d as c_int, rs.as_ptr(), std::ptr::null(), sb.as_ptr(), gamma);
    }
}

/// 양자화만 하고 sgemm 없이 u8+scale 반환 (공유 양자화용)
fn quant_only(
    x: &[f32], sl: usize, d: usize,
    nb: &mut Vec<f32>, ub: &mut Vec<u8>, sb: &mut Vec<f32>,
) {
    nb.resize(sl*d, 0.0);
    for t in 0..sl { rms_norm_no_affine(&x[t*d..(t+1)*d], &mut nb[t*d..(t+1)*d], 1e-5); }
    ub.resize(sl*d, 0); sb.resize(sl, 0.0);
    unsafe { batch_quantize_f32_to_u8(nb.as_ptr(), ub.as_mut_ptr(), sb.as_mut_ptr(), sl as c_int, d as c_int); }
}

/// 이미 양자화된 입력으로 sgemm만
fn sgemm_preq(
    ub: &[u8], sb: &[f32], sl: usize, in_d: usize, out_d: usize,
    w: &[i8], rs: &[i32], gamma: f32, out: &mut [f32],
) {
    unsafe {
        i8_sgemm(w.as_ptr(), ub.as_ptr(), out.as_mut_ptr(),
            out_d as c_int, sl as c_int, in_d as c_int, rs.as_ptr(), std::ptr::null(), sb.as_ptr(), gamma);
    }
}

pub fn benchmark_full_model(mt: &str, sl: usize, d: usize, warmup: usize, n_runs: usize) {
    let tot = sl*d;
    let hd = 32usize;
    let nh = d / hd;

    // mixing_proj_per_dir × 2dir × d² + ffn(3×d×dff) + norms(2×d) per layer
    // 128M 기준으로 n_layers 자동 계산
    let target_params: usize = 128_000_000;
    let overhead = 303 * d + 608 * d + d; // embedding + tag_head + final_norm

    // mamba2, mamba2_ds16, mamba2_ds128 등 지원
    let is_mamba2 = mt.starts_with("mamba2");
    let mt_base = if is_mamba2 { "mamba2" } else { mt };

    let (nmix, dff_ratio) = match mt_base {
        "fnet"   => (0usize, 8.0f64),  // mixing 없음 → FFN을 키움
        "tcn"    => (1, 2.66),          // 1 pointwise + depthwise(작음)
        "rwkv"   => (5, 2.66),          // r,k,v,o,g
        "retnet" => (5, 2.66),          // q,k,v,o,g
        "mamba"  => (0, 2.66),          // 특수 처리
        "mamba2" => (0, 2.66),          // 특수 처리 (SSD)
        "xlstm"  => (4, 2.66),          // i,f,z,o
        "mlstm"  => (5, 2.66),          // q,k,v,i,f
        _ => { eprintln!("알 수 없는: {}", mt); return; }
    };

    let dff = (d as f64 * dff_ratio) as usize;

    // mamba2 파싱: "mamba2_ds16" → ds=16, "mamba2_e15" → expand=1.5, "mamba2_e15_ds16" 등
    let m2_ds_override: usize = if mt.contains("ds") {
        let pos = mt.find("ds").unwrap() + 2;
        mt[pos..].split('_').next().unwrap_or("64").parse().unwrap_or(64)
    } else { 64 };
    let m2_expand_num: usize = if mt.contains("e15") { 3 } else { 2 }; // expand: 2=2x, 3=1.5x (num/2)
    let m2_expand_den: usize = if mt.contains("e15") { 2 } else { 1 };

    // per-layer 파라미터 계산
    let mix_params = match mt_base {
        "mamba" => 2 * (d*2*(2*d) + (2*d)*d + (d/16+2*16)*(2*d) + (d/16)*(2*d)), // 양방향: in+out+x+dt
        "mamba2" => {
            let di = d * m2_expand_num / m2_expand_den;
            let ds = m2_ds_override; let m2hd = 64usize;
            let m2nh = di / m2hd; let ng = 1usize; let dcv = 4usize;
            let d_conv_in = di + 2 * ng * ds;
            let d_in_proj = 2 * di + 2 * ng * ds + m2nh;
            2 * (d * d_in_proj + d_conv_in * (dcv + 1) + di + di * d + 3 * m2nh)
        }
        "tcn" => d * 7 * 6 + d * d, // 6 depthwise + 1 pointwise
        _ => 2 * nmix * d * d + d * d, // 양방향: 2 × N × d² + output d²
    };
    let ffn_params = 3 * d * dff; // gate + up + down
    let layer_params = mix_params + ffn_params + 2 * d; // + norms
    let nl = (target_params - overhead) / layer_params;

    // mixing fused projection 수 (output proj 포함 → +1)
    let total_mix_proj = if mt == "mamba" || is_mamba2 || mt == "fnet" { 0 } else { nmix };

    // 가중치
    let nw = vec![1.0f32; d];
    let (fmw, fmr) = if nmix>0 { make_dummy_i8(nmix*d, d) } else { (vec![], vec![]) };
    let (opw, opr) = make_dummy_i8(d, d);
    let (guw, gur) = make_dummy_i8(2*dff, d);
    let (dnw, dnr) = make_dummy_i8(d, dff);

    // Mamba 전용
    let di = d*2; let ds = 16usize; let dtr = d/16;
    let (miw, mir) = make_dummy_i8(2*di, d);
    let (mow, mor) = make_dummy_i8(d, di);
    let (mxw, mxr) = make_dummy_i8(dtr+2*ds, di);
    let (mdw, mdr) = make_dummy_i8(di, dtr);
    let ma = vec![-1.0f32; di*ds];
    let md_skip = vec![1.0f32; di];

    // Mamba-2 전용 (d_state, expand 파라미터에서 결정)
    let m2di = d * m2_expand_num / m2_expand_den;
    let m2hd = 64usize; let m2nh = m2di / m2hd; let m2ds = m2_ds_override; let m2ng = 1usize;
    let m2_d_conv_in = m2di + 2 * m2ng * m2ds;
    let m2_d_in_proj = 2 * m2di + 2 * m2ng * m2ds + m2nh;
    let (m2iw, m2ir) = make_dummy_i8(m2_d_in_proj, d);
    let (m2ow, m2or) = make_dummy_i8(d, m2di);
    let m2_decay: Vec<f32> = (0..m2nh).map(|i| 0.9 + 0.09 * i as f32 / m2nh as f32).collect();
    let m2_d_skip = vec![1.0f32; m2nh];
    let m2_conv_w = vec![0.1f32; m2_d_conv_in * 4];
    let m2_conv_b = vec![0.0f32; m2_d_conv_in];
    let m2_norm_w = vec![1.0f32; m2di];

    // 공통 scan 버퍼
    let wdecay = vec![-0.5f32; tot];
    let uparam = vec![0.1f32; d];
    let gammas: Vec<f32> = (0..nh).map(|i| 0.8+0.199*i as f32/7.0).collect();
    let tcnw = vec![0.1f32; d*7];

    let mut x: Vec<f32> = (0..tot).map(|i| (i as f32*0.01).sin()*0.1).collect();
    let mut nm = vec![0.0f32; tot];
    let mut pb = vec![0.0f32; sl*nmix.max(1)*d];
    let mut mo = vec![0.0f32; tot];
    let mut fo = vec![0.0f32; tot];
    let mut nb = vec![0.0f32; tot]; let mut ub = vec![0u8; tot]; let mut sb = vec![0.0f32; sl];
    let mut fb = vec![0.0f32; sl*2*dff.max(1)];
    // mamba bufs
    let mut mxz = vec![0.0f32; sl*2*di];
    let mut my = vec![0.0f32; sl*di];
    let mut mdt = vec![0.0f32; sl*di];
    let mut msp = vec![0.0f32; sl*(dtr+2*ds)];

    let run_layer = |x: &mut [f32], nm: &mut [f32], pb: &mut Vec<f32>, mo: &mut [f32], fo: &mut [f32],
                      nb: &mut Vec<f32>, ub: &mut Vec<u8>, sb: &mut Vec<f32>, fb: &mut Vec<f32>,
                      mxz: &mut [f32], my: &mut [f32], mdt: &mut [f32], msp: &mut [f32]| {
        // Mixing sub-layer
        rms_norm_batch_affine(x, &nw, 1e-6, sl, d, nm);

        match mt_base {
            "fnet" => { mo.copy_from_slice(nm); }
            "mamba" => {
                quant_sgemm(nm, sl, d, 2*di, &miw, &mir, 0.01, mxz, nb, ub, sb);
                quant_sgemm(&mxz[..sl*di], sl, di, dtr+2*ds, &mxw, &mxr, 0.01, msp, nb, ub, sb);
                quant_sgemm(&msp[..sl*dtr], sl, dtr, di, &mdw, &mdr, 0.01, mdt, nb, ub, sb);
                for v in mdt.iter_mut() { *v = softplus_scalar(*v); }
                let mut bssm = vec![0.0f32; sl*ds]; let mut cssm = vec![0.0f32; sl*ds];
                for t in 0..sl { for j in 0..ds {
                    bssm[t*ds+j] = msp[t*(dtr+2*ds)+dtr+j];
                    cssm[t*ds+j] = msp[t*(dtr+2*ds)+dtr+ds+j];
                }}
                let mut st = vec![0.0f32; di*ds];
                unsafe {
                    mamba_scan_avx2(mdt.as_ptr(), bssm.as_ptr(), cssm.as_ptr(), mxz.as_ptr(),
                        ma.as_ptr(), md_skip.as_ptr(), my.as_mut_ptr(), st.as_mut_ptr(), sl as c_int, di as c_int, ds as c_int);
                    st.fill(0.0);
                    mamba_scan_avx2(mdt.as_ptr(), bssm.as_ptr(), cssm.as_ptr(), mxz.as_ptr(),
                        ma.as_ptr(), md_skip.as_ptr(), my.as_mut_ptr(), st.as_mut_ptr(), sl as c_int, di as c_int, ds as c_int);
                }
                for i in 0..sl*di { my[i] *= silu_scalar(mxz[sl*di+i]); }
                quant_sgemm(my, sl, di, d, &mow, &mor, 0.01, mo, nb, ub, sb);
            }
            "mamba2" => {
                // in_proj: d → m2_d_in_proj
                let mut m2p = vec![0.0f32; sl * m2_d_in_proj];
                quant_sgemm(nm, sl, d, m2_d_in_proj, &m2iw, &m2ir, 0.01, &mut m2p, nb, ub, sb);

                // Split + conv1d + SiLU (간소화: xBC 연결 후 depthwise conv)
                let mut m2xbc = vec![0.0f32; sl * m2_d_conv_in];
                for t in 0..sl {
                    let src = t * m2_d_in_proj;
                    let dst = t * m2_d_conv_in;
                    m2xbc[dst..dst+m2di].copy_from_slice(&m2p[src..src+m2di]);
                    for j in 0..2*m2ng*m2ds {
                        m2xbc[dst+m2di+j] = m2p[src+2*m2di+j];
                    }
                }
                // causal depthwise conv1d (C 커널 — AVX2 벡터화)
                let mut m2conv = vec![0.0f32; sl * m2_d_conv_in];
                unsafe {
                    causal_conv1d_avx2(
                        m2xbc.as_ptr(), m2_conv_w.as_ptr(), m2_conv_b.as_ptr(),
                        m2conv.as_mut_ptr(),
                        sl as c_int, m2_d_conv_in as c_int, 4,
                    );
                }
                // Split x_conv (SiLU), B, C
                let mut m2x = vec![0.0f32; sl * m2di];
                let mut m2b = vec![0.0f32; sl * m2ng * m2ds];
                let mut m2c = vec![0.0f32; sl * m2ng * m2ds];
                let mut m2z = vec![0.0f32; sl * m2di];
                for t in 0..sl {
                    let cb = t * m2_d_conv_in;
                    let pb = t * m2_d_in_proj;
                    for j in 0..m2di { m2x[t*m2di+j] = silu_scalar(m2conv[cb+j]); }
                    for j in 0..m2ng*m2ds { m2b[t*m2ng*m2ds+j] = m2conv[cb+m2di+j]; }
                    for j in 0..m2ng*m2ds { m2c[t*m2ng*m2ds+j] = m2conv[cb+m2di+m2ng*m2ds+j]; }
                    for j in 0..m2di { m2z[t*m2di+j] = m2p[pb+m2di+j]; }
                }
                // Mamba-2 SSD scan (양방향: 2회)
                let mut m2y = vec![0.0f32; sl * m2di];
                let mut m2st = vec![0.0f32; m2nh * m2ds * m2hd];
                unsafe {
                    mamba2_scan_avx2(m2x.as_ptr(), m2b.as_ptr(), m2c.as_ptr(),
                        m2_decay.as_ptr(), m2_d_skip.as_ptr(),
                        m2y.as_mut_ptr(), m2st.as_mut_ptr(),
                        sl as c_int, m2nh as c_int, m2hd as c_int, m2ds as c_int, m2ng as c_int);
                    m2st.fill(0.0);
                    mamba2_scan_avx2(m2x.as_ptr(), m2b.as_ptr(), m2c.as_ptr(),
                        m2_decay.as_ptr(), m2_d_skip.as_ptr(),
                        m2y.as_mut_ptr(), m2st.as_mut_ptr(),
                        sl as c_int, m2nh as c_int, m2hd as c_int, m2ds as c_int, m2ng as c_int);
                }
                // RMSNorm + gate
                for t in 0..sl {
                    let b = t * m2di;
                    let mut sq = 0.0f32;
                    for j in 0..m2di { sq += m2y[b+j]*m2y[b+j]; }
                    let rms = (sq/m2di as f32 + 1e-5).sqrt().recip();
                    for j in 0..m2di { m2y[b+j] = m2y[b+j] * rms * m2_norm_w[j] * silu_scalar(m2z[b+j]); }
                }
                quant_sgemm(&m2y, sl, m2di, d, &m2ow, &m2or, 0.01, mo, nb, ub, sb);
            }
            "tcn" => {
                let mut acc = vec![0.0f32; tot]; let mut ct = vec![0.0f32; tot];
                for di_idx in 0..6 {
                    unsafe { depthwise_conv1d_avx2(nm.as_ptr(), tcnw.as_ptr(), std::ptr::null(),
                        ct.as_mut_ptr(), sl as c_int, d as c_int, 7, 1<<di_idx); }
                    for i in 0..tot { acc[i] += ct[i]; }
                }
                for v in acc.iter_mut() { *v = relu_scalar(*v); }
                quant_sgemm(&acc, sl, d, d, &opw, &opr, 0.01, mo, nb, ub, sb);
            }
            _ => {
                // Fused: 양자화 1회 → N개 proj sgemm
                let np = nmix;
                pb.resize(sl*np*d, 0.0);
                quant_only(nm, sl, d, nb, ub, sb);
                sgemm_preq(ub, sb, sl, d, np*d, &fmw, &fmr, 0.01, pb);

                // Scan (양방향)
                match mt_base {
                    "rwkv" => {
                        let mut st = vec![0.0f32; nh*hd*hd];
                        unsafe {
                            wkv6_scan_avx2(pb.as_ptr(), pb[tot..].as_ptr(), pb[2*tot..].as_ptr(),
                                wdecay.as_ptr(), uparam.as_ptr(), mo.as_mut_ptr(), st.as_mut_ptr(),
                                sl as c_int, nh as c_int, hd as c_int, d as c_int);
                            st.fill(0.0);
                            wkv6_scan_avx2(pb.as_ptr(), pb[tot..].as_ptr(), pb[2*tot..].as_ptr(),
                                wdecay.as_ptr(), uparam.as_ptr(), mo.as_mut_ptr(), st.as_mut_ptr(),
                                sl as c_int, nh as c_int, hd as c_int, d as c_int);
                        }
                        for i in 0..tot { mo[i] *= silu_scalar(pb[4*tot+i]); }
                    }
                    "retnet" => {
                        let mut st = vec![0.0f32; nh*hd*hd];
                        unsafe {
                            retention_scan_avx2(pb.as_ptr(), pb[tot..].as_ptr(), pb[2*tot..].as_ptr(),
                                gammas.as_ptr(), mo.as_mut_ptr(), st.as_mut_ptr(), sl as c_int, nh as c_int, hd as c_int);
                            st.fill(0.0);
                            retention_scan_avx2(pb.as_ptr(), pb[tot..].as_ptr(), pb[2*tot..].as_ptr(),
                                gammas.as_ptr(), mo.as_mut_ptr(), st.as_mut_ptr(), sl as c_int, nh as c_int, hd as c_int);
                        }
                        for i in 0..tot { mo[i] *= silu_scalar(pb[4*tot+i]); }
                    }
                    "xlstm" => {
                        let mut sc = vec![0.0f32; d]; let mut sn = vec![0.0f32; d];
                        unsafe {
                            slstm_scan_avx2(pb.as_ptr(), pb[tot..].as_ptr(), pb[2*tot..].as_ptr(), pb[3*tot..].as_ptr(),
                                mo.as_mut_ptr(), sc.as_mut_ptr(), sn.as_mut_ptr(), sl as c_int, d as c_int);
                            sc.fill(0.0); sn.fill(0.0);
                            slstm_scan_avx2(pb.as_ptr(), pb[tot..].as_ptr(), pb[2*tot..].as_ptr(), pb[3*tot..].as_ptr(),
                                mo.as_mut_ptr(), sc.as_mut_ptr(), sn.as_mut_ptr(), sl as c_int, d as c_int);
                        }
                    }
                    "mlstm" => {
                        let mut sc = vec![0.0f32; nh*hd*hd]; let mut sn = vec![0.0f32; nh*hd];
                        unsafe {
                            mlstm_scan_avx2(pb.as_ptr(), pb[tot..].as_ptr(), pb[2*tot..].as_ptr(),
                                pb[3*tot..].as_ptr(), pb[4*tot..].as_ptr(),
                                mo.as_mut_ptr(), sc.as_mut_ptr(), sn.as_mut_ptr(), sl as c_int, nh as c_int, hd as c_int);
                            sc.fill(0.0); sn.fill(0.0);
                            mlstm_scan_avx2(pb.as_ptr(), pb[tot..].as_ptr(), pb[2*tot..].as_ptr(),
                                pb[3*tot..].as_ptr(), pb[4*tot..].as_ptr(),
                                mo.as_mut_ptr(), sc.as_mut_ptr(), sn.as_mut_ptr(), sl as c_int, nh as c_int, hd as c_int);
                        }
                    }
                    _ => {}
                }
                // Output proj (fused 입력과 다른 데이터이므로 별도 양자화)
                quant_sgemm(mo, sl, d, d, &opw, &opr, 0.01, &mut pb[..tot], nb, ub, sb);
                mo.copy_from_slice(&pb[..tot]);
            }
        }
        for i in 0..tot { x[i] += mo[i]; }

        // FFN sub-layer (fused gate+up)
        rms_norm_batch_affine(x, &nw, 1e-6, sl, d, nm);
        quant_only(nm, sl, d, nb, ub, sb);
        fb.resize(sl*2*dff, 0.0);
        sgemm_preq(ub, sb, sl, d, 2*dff, &guw, &gur, 0.01, fb);
        for t in 0..sl { for i in 0..dff {
            let g = fb[t*2*dff+i]; let u = fb[t*2*dff+dff+i];
            fb[t*dff+i] = relu_scalar(g)*u;
        }}
        quant_sgemm(&fb[..sl*dff], sl, dff, d, &dnw, &dnr, 0.01, fo, nb, ub, sb);
        for i in 0..tot { x[i] += fo[i]; }
    };

    let mut run_full = |x: &mut Vec<f32>| {
        for _ in 0..nl {
            run_layer(x, &mut nm, &mut pb, &mut mo, &mut fo, &mut nb, &mut ub, &mut sb, &mut fb,
                &mut mxz, &mut my, &mut mdt, &mut msp);
        }
    };

    for _ in 0..warmup {
        x.iter_mut().enumerate().for_each(|(i,v)| *v = (i as f32*0.01).sin()*0.1);
        run_full(&mut x);
    }

    let mut lats = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        x.iter_mut().enumerate().for_each(|(i,v)| *v = (i as f32*0.01).sin()*0.1);
        let t0 = Instant::now();
        run_full(&mut x);
        lats.push(t0.elapsed().as_secs_f64()*1000.0);
    }

    let mut s = lats.clone(); s.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let med = s[s.len()/2];
    let avg = s.iter().sum::<f64>()/s.len() as f64;
    let p99 = s[((s.len() as f64*0.99) as usize).min(s.len()-1)];

    let total_p = overhead + nl * layer_params;
    println!("{:<10} d={:<3} {:>3}L {:>5.1}M {:>8.1}ms {:>7.2}ms/L {:>8.1}ms",
        mt, d, nl, total_p as f64 / 1e6, med, med/nl as f64, avg);
}

pub fn benchmark_all_full(sl: usize, d_model: usize, warmup: usize, n_runs: usize) {
    println!("=== DenseEditor 전체 모델 벤치마크 (128M, 실제 실행) ===");
    println!("seq_len={}, d_model={}, target=128M", sl, d_model);
    println!("fused projection + vectorized scan\n");
    println!("{:<10} {:>5} {:>5} {:>6} {:>10} {:>10} {:>10}",
        "Arch", "d", "Depth", "Params", "Median(ms)", "Per-L(ms)", "Mean(ms)");
    println!("{}", "-".repeat(62));
    for t in &["mamba", "mamba2", "mamba2_ds16", "mamba2_e15", "mamba2_e15_ds16"] {
        benchmark_full_model(t, sl, d_model, warmup, n_runs);
    }
}
