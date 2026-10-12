//! Window functions and millisecond/sample conversions.

/// Converts milliseconds to a whole number of samples (at least 1).
pub fn ms_to_samples(ms: f64, fs: u32) -> usize {
    ((ms * fs as f64 / 1000.0).round() as usize).max(1)
}

/// Power of two nearest (in log terms) to `ms` milliseconds, at least 64.
pub fn pow2_near_ms(ms: f64, fs: u32) -> usize {
    let target = ms * fs as f64 / 1000.0;
    let lo = (target.log2().floor() as u32).clamp(6, 20);
    let a = 1usize << lo;
    let b = a << 1;
    if (target / a as f64) <= (b as f64 / target) {
        a
    } else {
        b
    }
}

/// Periodic Hann window of length `n` (sums to a constant at hop `n/2`, `n/4`).
pub fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f64::consts::TAU * i as f64 / n as f64;
            (0.5 - 0.5 * x.cos()) as f32
        })
        .collect()
}

/// Hann window sampled at half-sample offsets: strictly positive everywhere, symmetric.
/// Used for WSOLA grains so that no grain contributes an exact-zero weight at its edge.
pub fn hann_offset(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f64::consts::TAU * (i as f64 + 0.5) / n as f64;
            (0.5 - 0.5 * x.cos()) as f32
        })
        .collect()
}

/// Square-root periodic Hann (analysis/synthesis pair for STFT).
pub fn sqrt_hann_periodic(n: usize) -> Vec<f32> {
    hann_periodic(n).into_iter().map(f32::sqrt).collect()
}

/// Raised-cosine fade from 0 to 1 over `n` samples, `t = (i + 0.5) / n`.
pub fn raised_cosine_ramp(i: usize, n: usize) -> f32 {
    if n == 0 {
        return 1.0;
    }
    let t = (i as f64 + 0.5) / n as f64;
    (0.5 - 0.5 * (std::f64::consts::PI * t).cos()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow2_choices() {
        assert_eq!(pow2_near_ms(25.0, 48_000), 1024);
        assert_eq!(pow2_near_ms(25.0, 44_100), 1024);
        assert_eq!(pow2_near_ms(25.0, 16_000), 512);
        assert_eq!(pow2_near_ms(25.0, 8_000), 256);
        assert_eq!(pow2_near_ms(25.0, 96_000), 2048);
    }

    #[test]
    fn hann_overlap_constant() {
        let n = 64;
        let w = hann_periodic(n);
        for i in 0..n / 4 {
            let s: f32 = (0..4).map(|k| w[i + k * n / 4]).sum();
            assert!((s - 2.0).abs() < 1e-5);
        }
        assert!(hann_offset(n).iter().all(|&v| v > 0.0));
    }
}
