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
