//! Band-limited resampling to an exact frame count (Kaiser-windowed sinc).
//!
//! Used by the pitch shifter: a signal stretched to `M` frames is resampled to `N`
//! frames and played at the original rate, scaling every frequency by `M / N`. Output
//! frame `j` reads the input at `t_j = (j + 1/2) · M/N − 1/2`, so both ends line up.
//! When `M > N` the cutoff drops to the output Nyquist to prevent aliasing.

/// Zero crossings of the sinc on each side of the kernel centre.
const ZERO_CROSSINGS: usize = 16;
/// Table entries per zero crossing (linear interpolation between them).
const OVERSAMPLE: usize = 512;
/// Kaiser β: ≈ 90 dB stopband attenuation.
const KAISER_BETA: f64 = 9.0;
/// Passband edge as a fraction of the lower of the two Nyquist frequencies.
const PASSBAND: f64 = 0.94;

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    for k in 1..64 {
        term *= q / (k * k) as f64;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// `sinc(u) · kaiser(u / Z)` for `u ∈ [0, Z]`, `OVERSAMPLE` entries per unit (+1 guard).
fn kernel_table() -> Vec<f64> {
    let n = ZERO_CROSSINGS * OVERSAMPLE;
    let norm = bessel_i0(KAISER_BETA);
    (0..=n + 1)
        .map(|i| {
            let u = i as f64 / OVERSAMPLE as f64;
            if u >= ZERO_CROSSINGS as f64 {
                return 0.0;
            }
            let sinc = if i == 0 {
                1.0
            } else {
                let a = std::f64::consts::PI * u;
                a.sin() / a
            };
            let r = u / ZERO_CROSSINGS as f64;
            sinc * bessel_i0(KAISER_BETA * (1.0 - r * r).max(0.0).sqrt()) / norm
        })
        .collect()
}

/// Input position read by output frame `j` when mapping `m` input frames onto `n`.
#[inline]
fn source_pos(j: usize, m: usize, n: usize) -> f64 {
    (j as f64 + 0.5) * m as f64 / n as f64 - 0.5
}

/// Reusable resampler for one `m → n` mapping (the kernel table is shared by all
/// channels).
pub struct Resampler {
    table: Vec<f64>,
    m: usize,
    n: usize,
    /// Cutoff in cycles per input sample, times two (`2 f_c`).
    two_fc: f64,
    /// Kernel half-width in input samples.
    half: f64,
    /// Worker threads for [`Resampler::process`] (output frames are independent).
    threads: usize,
}

impl Resampler {
    pub fn new(m: usize, n: usize) -> Self {
        let rho = m as f64 / n.max(1) as f64;
        let two_fc = PASSBAND * (1.0 / rho).min(1.0);
        Self {
            table: kernel_table(),
            m,
            n,
            two_fc,
            half: ZERO_CROSSINGS as f64 / two_fc,
            threads: 1,
        }
    }

    /// Uses up to `threads` workers (`0` = all cores). The result does not depend on it.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = crate::par::threads(threads);
        self
    }

    #[inline]
    fn kernel(&self, tau: f64) -> f64 {
        let u = (tau * self.two_fc).abs() * OVERSAMPLE as f64;
        let i = u as usize;
        if i >= ZERO_CROSSINGS * OVERSAMPLE {
            return 0.0;
        }
        let f = u - i as f64;
        self.two_fc * (self.table[i] + f * (self.table[i + 1] - self.table[i]))
    }

    /// Resamples `x` (length `m`) to exactly `n` frames. Samples outside `x` count as 0.
    /// `progress(done, total)` is called every few thousand output frames.
    pub fn process(&self, x: &[f32], progress: &mut dyn FnMut(usize, usize)) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.m);
        let mut y = vec![0f32; self.n];
        if self.m == 0 {
            return y;
        }
        let last = self.m as i64 - 1;
        let frame = |j: usize| -> f32 {
            let t = source_pos(j, self.m, self.n);
            let a = ((t - self.half).ceil() as i64).max(0);
            let b = ((t + self.half).floor() as i64).min(last);
            let mut acc = 0f64;
            for i in a..=b {
                acc += x[i as usize] as f64 * self.kernel(t - i as f64);
            }
            acc as f32
        };
        // Blocks keep progress reporting on the calling thread.
        const BLOCK: usize = 1 << 16;
        for (bi, block) in y.chunks_mut(BLOCK).enumerate() {
            let j0 = bi * BLOCK;
            let parts = crate::par::chunks(block.len(), self.threads, 4096, |r| {
                r.map(|k| frame(j0 + k)).collect::<Vec<f32>>()
            });
            for (o, v) in block.iter_mut().zip(parts.into_iter().flatten()) {
                *o = v;
            }
            progress(j0, self.n);
        }
        progress(self.n, self.n);
        y
    }
}

/// Linear-interpolation resampling for control signals (masks), same time alignment as
/// [`Resampler`].
pub fn resample_linear(x: &[f32], n: usize) -> Vec<f32> {
    let m = x.len();
    if m == 0 {
        return vec![0.0; n];
    }
    (0..n)
        .map(|j| {
            let t = source_pos(j, m, n).clamp(0.0, (m - 1) as f64);
            let i = t as usize;
            let f = (t - i as f64) as f32;
            let b = x[(i + 1).min(m - 1)];
            x[i] + f * (b - x[i])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, cycles_per_sample: f64) -> Vec<f32> {
        (0..n)
            .map(|i| (std::f64::consts::TAU * cycles_per_sample * i as f64).sin() as f32 * 0.5)
            .collect()
    }

    /// Amplitude of `f` (cycles/sample) in `y`, ignoring `skip` frames at each end.
    fn amplitude(y: &[f32], f: f64, skip: usize) -> f64 {
        let (mut c, mut s) = (0.0, 0.0);
        let core = &y[skip..y.len() - skip];
        for (i, &v) in core.iter().enumerate() {
            let a = std::f64::consts::TAU * f * (i + skip) as f64;
            c += v as f64 * a.cos();
            s += v as f64 * a.sin();
        }
        2.0 * (c * c + s * s).sqrt() / core.len() as f64
    }

    #[test]
    fn exact_lengths() {
        for (m, n) in [
            (1, 1),
            (1, 7),
            (7, 1),
            (1000, 999),
            (4800, 9600),
            (9600, 4800),
        ] {
            let x = vec![0.1f32; m];
            assert_eq!(Resampler::new(m, n).process(&x, &mut |_, _| {}).len(), n);
            assert_eq!(resample_linear(&x, n).len(), n);
        }
    }

    #[test]
    fn tone_frequency_scales_by_m_over_n() {
        // 1 kHz at 48 kHz, 6000 → 4000 frames: plays back at 1.5 kHz.
        let (m, n) = (6000, 4000);
        let f = 1000.0 / 48_000.0;
        let y = Resampler::new(m, n).process(&tone(m, f), &mut |_, _| {});
        let f_out = f * m as f64 / n as f64;
        assert!((amplitude(&y, f_out, 200) - 0.5).abs() < 0.01);
        assert!(amplitude(&y, f, 200) < 0.01);
    }

    #[test]
    fn content_above_output_nyquist_is_removed() {
        // 0.4 cycles/sample, compressed 2:1 would alias to 0.2 without the low-pass.
        let (m, n) = (8000, 4000);
        let y = Resampler::new(m, n).process(&tone(m, 0.4), &mut |_, _| {});
        let peak = y[100..n - 100].iter().fold(0f32, |a, &v| a.max(v.abs()));
        assert!(peak < 1e-3, "aliased residue {peak}");
    }

    #[test]
    fn zeros_stay_zero() {
        let y = Resampler::new(500, 700).process(&[0.0; 500], &mut |_, _| {});
        assert!(y.iter().all(|&v| v == 0.0));
    }
}
