//! Bit-exact `int16 x int16 -> int64` by int8 byte decomposition.
//!
//! Port of `rocket_matmul_int16_exact` in `rocket_matmul.c`. The RK3588 NPU has no
//! native int16 matmul (its only integer matmul mode is int8 -> int32), so the C
//! library splits every int16 into two signed bytes and runs **four** proven int8
//! matmuls, recombining in int64:
//!
//! ```text
//! x  = xh*256 + xl            balanced signed split, xh, xl in [-128, 127]
//! xl = ((x + 128) & 0xFF) - 128
//! xh = (x - xl) >> 8
//! C  = 65536*(Ah·Bh) + 256*(Ah·Bl + Al·Bh) + (Al·Bl)
//! ```
//!
//! Two signed bytes span `[-32896, 32639]`, so inputs are clamped to
//! `[-32768, 32639]` first; inside that domain every value round-trips exactly and the
//! recombined result is the exact int64 product sum (no saturation anywhere).
//!
//! Like the C, this makes four full [`one_shot_int8`] calls, each with its own
//! transient BO set — the arithmetic is integer-exact either way, so sharing one dense
//! context would buy wall-clock, not fidelity, and the C does not share one.

use crate::Device;
use crate::error::{Error, Result};

use super::int8_dense::one_shot_int8;

/// The signed-byte split of one clamped int16 value: `(hi, lo)`.
#[inline]
fn split_one(x: i16) -> (i8, i8) {
    let x = (x as i32).clamp(-32768, 32639);
    let l = ((x + 128) & 0xFF) - 128;
    (((x - l) >> 8) as i8, l as i8)
}

/// Split `[i16]` into `(hi, lo)` byte planes, clamping as the C does.
fn split(x: &[i16]) -> (Vec<i8>, Vec<i8>) {
    let mut hi = Vec::with_capacity(x.len());
    let mut lo = Vec::with_capacity(x.len());
    for &v in x {
        let (h, l) = split_one(v);
        hi.push(h);
        lo.push(l);
    }
    (hi, lo)
}

/// The C recombination, in int64.
#[inline]
fn combine(hh: i32, hl: i32, lh: i32, ll: i32) -> i64 {
    65536 * (hh as i64) + 256 * ((hl as i64) + (lh as i64)) + (ll as i64)
}

/// `C[M,N] = A[M,K] * B[N,K]^T`, int16 x int16 -> exact int64 (the C
/// `rocket_matmul_int16_exact`).
///
/// Inputs are clamped to `[-32768, 32639]` (the signed-byte domain) and the result is
/// bit-exact within it. `M == 1` is padded to a 4-row tile and row 0 is returned, like
/// every other one-shot path; other ragged `M` are rejected by the int8 planner
/// (`M % 4`), and so are `K % 32` / `N % 32`.
///
/// Cost is four int8 matmuls (~4x int8): completeness, not speed.
pub fn one_shot_int16_exact(
    dev: &Device,
    m: usize,
    k: usize,
    n: usize,
    a: &[i16],
    b: &[i16],
    c: &mut [i64],
) -> Result<()> {
    if m == 1 {
        if a.len() < k || b.len() < n * k || c.len() < n {
            return Err(Error::shape("int16/buffers"));
        }
        let mut ap = vec![0i16; 4 * k];
        ap[..k].copy_from_slice(a);
        let mut cp = vec![0i64; 4 * n];
        one_shot_int16_exact(dev, 4, k, n, &ap, b, &mut cp)?;
        c[..n].copy_from_slice(&cp[..n]);
        return Ok(());
    }
    if a.len() < m * k || b.len() < n * k || c.len() < m * n {
        return Err(Error::shape("int16/buffers"));
    }
    let (ah, al) = split(&a[..m * k]);
    let (bh, bl) = split(&b[..n * k]);

    let mut hh = vec![0i32; m * n];
    let mut hl = vec![0i32; m * n];
    let mut lh = vec![0i32; m * n];
    let mut ll = vec![0i32; m * n];
    one_shot_int8(dev, m, k, n, &ah, &bh, &mut hh)?;
    one_shot_int8(dev, m, k, n, &ah, &bl, &mut hl)?;
    one_shot_int8(dev, m, k, n, &al, &bh, &mut lh)?;
    one_shot_int8(dev, m, k, n, &al, &bl, &mut ll)?;

    for i in 0..m * n {
        c[i] = combine(hh[i], hl[i], lh[i], ll[i]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic LCG, so the decomposition test needs no dev-dependency.
    fn rnd(seed: &mut u64) -> i16 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as u16) as i16
    }

    #[test]
    fn split_is_exact_inside_the_signed_byte_domain() {
        let mut vals = vec![
            -32768i16, -32767, -30000, -257, -256, -255, -129, -128, -127, -1, 0, 1, 127, 128, 255,
            256, 257, 30000, 32638, 32639,
        ];
        let mut seed = 7u64;
        vals.extend((0..2000).map(|_| rnd(&mut seed)));
        for x in vals {
            // Two signed bytes span [-32896, 32639]; above that the value is clamped,
            // and inside the domain the split is an exact re-encoding.
            let want = (x as i32).clamp(-32768, 32639);
            let (h, l) = split_one(x);
            assert_eq!(h as i32 * 256 + l as i32, want, "x={x}");
            assert!((-128..=127).contains(&(h as i32)), "hi out of range: {x}");
            assert!((-128..=127).contains(&(l as i32)), "lo out of range: {x}");
        }
    }

    #[test]
    fn the_top_codes_clamp() {
        // Two signed bytes reach +32639, so everything above it clamps.
        assert_eq!(split_one(32639), (127, 127));
        assert_eq!(split_one(32640), (127, 127));
        assert_eq!(split_one(i16::MAX), (127, 127));
        assert_eq!(split_one(i16::MIN), (-128, 0));
    }

    #[test]
    fn the_decomposition_reproduces_the_exact_product() {
        // A 5x64x3 problem: decompose both operands, run the four products on the host
        // in i64, recombine, and compare with the direct exact dot product.
        let (m, k, n) = (5usize, 64usize, 3usize);
        let mut seed = 0xdead_beef_1234_5678u64;
        let a: Vec<i16> = (0..m * k).map(|_| rnd(&mut seed)).collect();
        let b: Vec<i16> = (0..n * k).map(|_| rnd(&mut seed)).collect();
        let (ah, al) = split(&a);
        let (bh, bl) = split(&b);
        for i in 0..m {
            for j in 0..n {
                let mut want = 0i64;
                let mut hh = 0i64;
                let mut hl = 0i64;
                let mut lh = 0i64;
                let mut ll = 0i64;
                for p in 0..k {
                    let x = (a[i * k + p] as i32).clamp(-32768, 32639) as i64;
                    let y = (b[j * k + p] as i32).clamp(-32768, 32639) as i64;
                    want += x * y;
                    hh += ah[i * k + p] as i64 * bh[j * k + p] as i64;
                    hl += ah[i * k + p] as i64 * bl[j * k + p] as i64;
                    lh += al[i * k + p] as i64 * bh[j * k + p] as i64;
                    ll += al[i * k + p] as i64 * bl[j * k + p] as i64;
                }
                assert_eq!(combine(hh as i32, hl as i32, lh as i32, ll as i32), want);
            }
        }
    }
}
