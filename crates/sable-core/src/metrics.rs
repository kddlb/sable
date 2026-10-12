//! Objective measurements shared by tests, examples and the quality report.

use crate::features::{FeatureConfig, PitchEstimator};

/// Median F0 (Hz) over `[a, b)` using 10-ms steps, counting only frames with
/// periodicity above `min_periodicity`. Returns `None` if no frame qualifies.
pub fn median_f0(x: &[f32], fs: u32, a: usize, b: usize, min_periodicity: f32) -> Option<f64> {
    let mut pe = PitchEstimator::new(fs, &FeatureConfig::default());
    let step = (fs as usize / 100).max(1);
    let mut v = Vec::new();
    let mut c = a;
    while c < b.min(x.len()) {
        let (p, f0, _) = pe.estimate(x, c);
        if p >= min_periodicity && f0 > 0.0 {
            v.push(f0 as f64);
        }
        c += step;
    }
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(v[v.len() / 2])
}

/// RMS of `x[a..b]`.
pub fn rms(x: &[f32], a: usize, b: usize) -> f64 {
    let b = b.min(x.len());
    if b <= a {
        return 0.0;
    }
    (x[a..b].iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / (b - a) as f64).sqrt()
}

/// Welch power spectral density of `x[a..b]` (Hann, 50 % overlap), one value per bin.
pub fn welch_psd(x: &[f32], a: usize, b: usize, n_fft: usize) -> Vec<f64> {
    welch_cross(x, x, a, b, n_fft).0
}

/// Welch auto-spectra of `x`, `y` and cross-spectrum `Y conj X` over `[a, b)`.
/// Returns `(Pxx, Pyy, (re, im) of Pyx)`.
#[allow(clippy::type_complexity)]
pub fn welch_cross(
    x: &[f32],
    y: &[f32],
    a: usize,
    b: usize,
    n_fft: usize,
) -> (Vec<f64>, Vec<f64>, Vec<(f64, f64)>) {
    let bins = n_fft / 2 + 1;
    let win = crate::windows::hann_periodic(n_fft);
    let mut fft = crate::fft::RealFft::new(n_fft);
    let mut pxx = vec![0f64; bins];
    let mut pyy = vec![0f64; bins];
    let mut pyx = vec![(0f64, 0f64); bins];
    let mut count = 0;
    let mut s = a;
    while s + n_fft <= b.min(x.len()).min(y.len()) {
        let fx = fft.forward_windowed(&x[s..s + n_fft], &win).to_vec();
        let fy = fft.forward_windowed(&y[s..s + n_fft], &win).to_vec();
        for k in 0..bins {
            pxx[k] += fx[k].norm_sqr() as f64;
            pyy[k] += fy[k].norm_sqr() as f64;
            pyx[k].0 += (fy[k].re * fx[k].re + fy[k].im * fx[k].im) as f64;
            pyx[k].1 += (fy[k].im * fx[k].re - fy[k].re * fx[k].im) as f64;
        }
        count += 1;
        s += n_fft / 2;
    }
    let c = count.max(1) as f64;
    for k in 0..bins {
        pxx[k] /= c;
        pyy[k] /= c;
        pyx[k].0 /= c;
        pyx[k].1 /= c;
    }
    (pxx, pyy, pyx)
}

/// Level (dB) in third-octave bands between `lo_hz` and `hi_hz`; returns `(center, dB)`.
pub fn band_levels_db(psd: &[f64], fs: u32, lo_hz: f64, hi_hz: f64) -> Vec<(f64, f64)> {
    let n_fft = (psd.len() - 1) * 2;
    let bin_hz = fs as f64 / n_fft as f64;
    let mut out = Vec::new();
    let mut fc = lo_hz;
    while fc * 2f64.powf(1.0 / 6.0) <= hi_hz {
        let (a, b) = (fc * 2f64.powf(-1.0 / 6.0), fc * 2f64.powf(1.0 / 6.0));
        let (ka, kb) = (
            (a / bin_hz).ceil() as usize,
            ((b / bin_hz).floor() as usize).min(psd.len() - 1),
        );
        if kb >= ka {
            let p: f64 = psd[ka..=kb].iter().sum();
            out.push((fc, 10.0 * (p + 1e-30).log10()));
        }
        fc *= 2f64.powf(1.0 / 3.0);
    }
    out
}

/// Power-weighted spectral centroid (Hz).
pub fn centroid_hz(psd: &[f64], fs: u32) -> f64 {
    let n_fft = (psd.len() - 1) * 2;
    let bin_hz = fs as f64 / n_fft as f64;
    let (mut num, mut den) = (0.0, 0.0);
    for (k, &p) in psd.iter().enumerate() {
        num += p * k as f64 * bin_hz;
        den += p;
    }
    if den > 0.0 { num / den } else { 0.0 }
}

/// Mean magnitude-squared coherence over bins in `[lo_hz, hi_hz]`.
pub fn mean_coherence(
    l: &[f32],
    r: &[f32],
    a: usize,
    b: usize,
    n_fft: usize,
    fs: u32,
    (lo_hz, hi_hz): (f64, f64),
) -> f64 {
    let (pll, prr, prl) = welch_cross(l, r, a, b, n_fft);
    let bin_hz = fs as f64 / n_fft as f64;
    let (mut s, mut n) = (0.0, 0);
    for k in 0..pll.len() {
        let f = k as f64 * bin_hz;
        if f < lo_hz || f > hi_hz || pll[k] <= 0.0 || prr[k] <= 0.0 {
            continue;
        }
        s += (prl[k].0.powi(2) + prl[k].1.powi(2)) / (pll[k] * prr[k]);
        n += 1;
    }
    if n > 0 { s / n as f64 } else { 0.0 }
}

/// Largest normalized autocorrelation of `x[a..b]` for lags in `[min_lag, max_lag]`:
/// a high value in noise means audible repetition of grains.
pub fn max_autocorr(x: &[f32], a: usize, b: usize, min_lag: usize, max_lag: usize) -> f64 {
    let seg: Vec<f64> = x[a..b.min(x.len())].iter().map(|&v| v as f64).collect();
    let e0: f64 = seg.iter().map(|v| v * v).sum();
    if e0 <= 0.0 {
        return 0.0;
    }
    let mut best = 0f64;
    for lag in min_lag..=max_lag.min(seg.len().saturating_sub(1)) {
        let n = seg.len() - lag;
        let (mut num, mut ea, mut eb) = (0.0, 0.0, 0.0);
        for i in 0..n {
            num += seg[i] * seg[i + lag];
            ea += seg[i] * seg[i];
            eb += seg[i + lag] * seg[i + lag];
        }
        if ea > 0.0 && eb > 0.0 {
            best = best.max(num / (ea * eb).sqrt());
        }
    }
    best
}
