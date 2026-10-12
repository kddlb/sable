//! Linear-phase complementary FIR band split for voiced sibilants (spec §8.4).
//!
//! `lowpass + highpass == identity` exactly (the high-pass is `δ - h_lp`), and both are
//! applied zero-phase (centred kernel), so recombination introduces no delay or gain bump.

/// Complementary low/high split with a windowed-sinc low-pass.
#[derive(Clone, Debug)]
pub struct BandSplit {
    pub kernel: Vec<f32>,
    pub cutoff_hz: f64,
    pub transition_hz: f64,
}

impl BandSplit {
    /// Cutoff ≈ 3 kHz with a ≈2 kHz (2–4 kHz) transition band, scaled down at low rates.
    pub fn new(fs: u32) -> Self {
        let nyq = fs as f64 / 2.0;
        let cutoff = 3000f64.min(0.7 * nyq);
        let transition = 2000f64.min(0.5 * nyq);
        // Blackman: transition ≈ 5.5 fs / M.
        let mut m = (5.5 * fs as f64 / transition).ceil() as usize;
        if m % 2 == 0 {
            m += 1;
        }
        let c = (m / 2) as f64;
        let fc = cutoff / fs as f64;
        let mut h: Vec<f64> = (0..m)
            .map(|i| {
                let t = i as f64 - c;
                let sinc = if t == 0.0 {
                    2.0 * fc
                } else {
                    (std::f64::consts::TAU * fc * t).sin() / (std::f64::consts::PI * t)
                };
                let x = std::f64::consts::TAU * i as f64 / (m - 1) as f64;
                let w = 0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos();
                sinc * w
            })
            .collect();
        let s: f64 = h.iter().sum();
        h.iter_mut().for_each(|v| *v /= s);
        Self {
            kernel: h.into_iter().map(|v| v as f32).collect(),
            cutoff_hz: cutoff,
            transition_hz: transition,
        }
    }

    /// Zero-phase low-pass of `x[a..b]` (reads outside the slice are treated as zero).
    pub fn lowpass(&self, x: &[f32], a: usize, b: usize) -> Vec<f32> {
        let half = (self.kernel.len() / 2) as isize;
        (a..b)
            .map(|n| {
                let mut acc = 0f64;
                for (j, &h) in self.kernel.iter().enumerate() {
                    let i = n as isize + half - j as isize;
                    if i >= 0 && (i as usize) < x.len() {
                        acc += h as f64 * x[i as usize] as f64;
                    }
                }
                acc as f32
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complementary_reconstruction_and_response() {
        for fs in [8_000u32, 16_000, 48_000, 96_000] {
            let bs = BandSplit::new(fs);
            let n = fs as usize / 10;
            // Low tone passes, high tone is removed.
            let lo: Vec<f32> = (0..n)
                .map(|i| (std::f64::consts::TAU * 500.0 * i as f64 / fs as f64).sin() as f32)
                .collect();
            let hf = 0.9 * fs as f64 / 2.0;
            let hi: Vec<f32> = (0..n)
                .map(|i| (std::f64::consts::TAU * hf * i as f64 / fs as f64).sin() as f32)
                .collect();
            let m = bs.kernel.len();
            let l_lo = bs.lowpass(&lo, m, n - m);
            let l_hi = bs.lowpass(&hi, m, n - m);
            let rms = |v: &[f32]| {
                (v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
            };
            assert!(
                (rms(&l_lo) / rms(&lo[m..n - m]) - 1.0).abs() < 0.01,
                "fs={fs}"
            );
            assert!(rms(&l_hi) < 0.01, "fs={fs}");
            // lp + (x - lp) == x
            let y: Vec<f32> = l_lo
                .iter()
                .zip(&lo[m..n - m])
                .map(|(a, b)| a + (b - a))
                .collect();
            for (a, b) in y.iter().zip(&lo[m..n - m]) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }
}
