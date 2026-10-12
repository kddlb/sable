//! Spectral-envelope pre-warp for formant-preserving pitch shifting.
//!
//! Resampling a signal by `1/s` multiplies every frequency by `s`, so the spectral
//! envelope `H(f)` (formants, sibilant colour) moves to `H(f / s)`. Before resampling,
//! each STFT frame is filtered by `G(f) = H(s·f) / H(f)`; after resampling the envelope
//! is back at `H(f)` while the harmonics sit at `s` times their original frequencies.
//!
//! `H` is the **true envelope** (Röbel & Rodet): iterated cepstral smoothing that rides
//! the harmonic peaks instead of averaging between them. The lifter order is half the
//! pitch period of the speaker's upper F0 range, measured by the analysis stage. One
//! envelope is estimated from the channel mean and the same gain is applied to every
//! channel, so stereo inter-channel relations are untouched.

use serde::Serialize;

use crate::fft::{Complex, RealFft};
use crate::windows::{hann_periodic, pow2_near_ms};

/// STFT frame length; long enough for three periods of a 75 Hz voice.
const FRAME_MS: f64 = 40.0;
/// True-envelope iterations (it usually converges in 3–6).
const MAX_ITERATIONS: usize = 12;
/// Stop iterating when the envelope is within this many dB of every spectral peak.
const CONVERGENCE_DB: f64 = 1.5;
/// Envelope correction limit, so near-silent or band-limited bins cannot explode.
const MAX_GAIN_DB: f64 = 30.0;
/// Spectral floor relative to the frame's peak bin, before taking logs.
const FLOOR_DB: f64 = -100.0;

/// What the correction did, for the diagnostic report.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FormantStats {
    pub frame_len: usize,
    pub hop: usize,
    pub frames: usize,
    /// F0 that set the lifter (upper voiced range of the input).
    pub f0_ref_hz: f64,
    /// Cepstral lifter cut-off in milliseconds of quefrency.
    pub lifter_ms: f64,
    pub mean_iterations: f64,
}

/// Cepstral smoothing of a log-magnitude spectrum, keeping quefrencies `<= order`.
fn cepstral_smooth(
    fft: &mut RealFft,
    log_mag: &[f64],
    order: usize,
    spec: &mut [Complex],
    out: &mut [f64],
) {
    for (s, &l) in spec.iter_mut().zip(log_mag) {
        *s = Complex::new(l as f32, 0.0);
    }
    let k = fft.len();
    let mut ceps = fft.inverse(spec).to_vec();
    for (q, c) in ceps.iter_mut().enumerate() {
        let quef = q.min(k - q);
        if quef > order {
            *c = 0.0;
        } else if quef == order {
            *c *= 0.5;
        }
    }
    for (o, s) in out.iter_mut().zip(fft.forward(&ceps)) {
        *o = s.re as f64;
    }
}

/// Per-worker FFT and spectrum buffers for [`prewarp`].
struct FrameScratch {
    fft: RealFft,
    frame: Vec<f32>,
    chan_spec: Vec<Vec<Complex>>,
    log_mag: Vec<f64>,
    a: Vec<f64>,
    env: Vec<f64>,
    scratch: Vec<Complex>,
    mono_mag: Vec<f64>,
}

impl FrameScratch {
    fn new(k: usize, nch: usize) -> Self {
        let bins = k / 2 + 1;
        Self {
            fft: RealFft::new(k),
            frame: vec![0f32; k],
            chan_spec: vec![vec![Complex::new(0.0, 0.0); bins]; nch],
            log_mag: vec![0f64; bins],
            a: vec![0f64; bins],
            env: vec![0f64; bins],
            scratch: vec![Complex::new(0.0, 0.0); bins],
            mono_mag: vec![0f64; bins],
        }
    }
}

/// Filters `channels` in place so their envelope reads `H(s·f)` at `f`.
///
/// Frames are corrected on up to `threads` workers in batches; each batch is then
/// overlap-added in frame order, so the result does not depend on the thread count.
pub fn prewarp(
    channels: &mut [Vec<f32>],
    fs: u32,
    s: f64,
    f0_ref_hz: f64,
    threads: usize,
    progress: &mut dyn FnMut(usize, usize),
) -> FormantStats {
    let n = pow2_near_ms(FRAME_MS, fs);
    let hop = n / 4;
    let k = 2 * n;
    let bins = k / 2 + 1;
    let order = ((fs as f64 / (2.0 * f0_ref_hz)).floor() as usize).clamp(4, n / 2);
    let len = channels.first().map_or(0, Vec::len);
    let nch = channels.len();
    let mut stats = FormantStats {
        frame_len: n,
        hop,
        frames: 0,
        f0_ref_hz,
        lifter_ms: order as f64 * 1000.0 / fs as f64,
        mean_iterations: 0.0,
    };
    if len == 0 || nch == 0 {
        return stats;
    }
    let win = hann_periodic(n);
    // Periodic Hann at hop n/4 overlap-adds to exactly 2.
    let ola_gain = 0.5f32;
    let mut out = vec![vec![0f32; len]; nch];
    let floor_ratio = 10f64.powf(FLOOR_DB / 20.0);
    let conv = CONVERGENCE_DB / 20.0 * std::f64::consts::LN_10;
    let max_gain = MAX_GAIN_DB / 20.0 * std::f64::consts::LN_10;
    let mut total_iters = 0usize;

    // Frame centres from −n/2 to len + n/2 so every sample gets the full window sum.
    let first = -(n as i64 / 2);
    let total = (len as i64 + n as i64 / 2 - first) as usize / hop + 1;
    let centre = |f: usize| first + (f * hop) as i64;
    let input: &[Vec<f32>] = channels;
    // Corrected `2n`-sample frame per channel and the iteration count, or `None` for a
    // silent frame.
    let correct = |f: usize, w: &mut FrameScratch| -> Option<(Vec<Vec<f32>>, usize)> {
        let c = centre(f);
        // Windowed frame in the middle of a 2n buffer: index i ↔ sample c − n + i.
        let mut any = false;
        w.mono_mag.fill(0.0);
        for (x, spec) in input.iter().zip(w.chan_spec.iter_mut()) {
            w.frame.fill(0.0);
            for (w_i, &wv) in win.iter().enumerate() {
                let t = c - (n as i64) / 2 + w_i as i64;
                if (0..len as i64).contains(&t) {
                    let v = x[t as usize] * wv;
                    any |= v != 0.0;
                    w.frame[n / 2 + w_i] = v;
                }
            }
            spec.copy_from_slice(w.fft.forward(&w.frame));
        }
        if !any {
            return None;
        }
        // Envelope of the channel mean.
        for (b, m) in w.mono_mag.iter_mut().enumerate() {
            let sum: Complex = w.chan_spec.iter().map(|s| s[b]).sum();
            *m = (sum.norm() / nch as f32) as f64;
        }
        let peak = w.mono_mag.iter().fold(0f64, |p, &v| p.max(v));
        let floor = (peak * floor_ratio).max(1e-30);
        for (l, &m) in w.log_mag.iter_mut().zip(&w.mono_mag) {
            *l = (m + floor).ln();
        }
        w.a.copy_from_slice(&w.log_mag);
        let mut iters = 0;
        loop {
            cepstral_smooth(&mut w.fft, &w.a, order, &mut w.scratch, &mut w.env);
            iters += 1;
            let gap = w
                .log_mag
                .iter()
                .zip(&w.env)
                .fold(f64::NEG_INFINITY, |g, (&l, &e)| g.max(l - e));
            if gap < conv || iters >= MAX_ITERATIONS {
                break;
            }
            for (ai, &e) in w.a.iter_mut().zip(&w.env) {
                *ai = ai.max(e);
            }
        }
        // Gain G(f) = H(s·f) / H(f), clamped; bins mapped past Nyquist read H(Nyquist).
        let env = &w.env;
        for b in 0..bins {
            let x = b as f64 * s;
            let target = if x >= (bins - 1) as f64 {
                env[bins - 1]
            } else {
                let i = x as usize;
                let fr = x - i as f64;
                env[i] + fr * (env[i + 1] - env[i])
            };
            let g = (target - env[b]).clamp(-max_gain, max_gain).exp() as f32;
            for spec in w.chan_spec.iter_mut() {
                spec[b] *= g;
            }
        }
        let ys = (0..nch)
            .map(|ch| w.fft.inverse(&w.chan_spec[ch]).to_vec())
            .collect();
        Some((ys, iters))
    };

    // Batches bound the memory held for corrected frames.
    let batch = threads.max(1) * 32;
    let mut f0 = 0;
    while f0 < total {
        let f1 = (f0 + batch).min(total);
        let parts = crate::par::chunks(f1 - f0, threads, 8, |r| {
            let mut w = FrameScratch::new(k, nch);
            r.map(|i| correct(f0 + i, &mut w)).collect::<Vec<_>>()
        });
        for (f, res) in (f0..f1).zip(parts.into_iter().flatten()) {
            let Some((ys, iters)) = res else { continue };
            total_iters += iters;
            stats.frames += 1;
            let base = centre(f) - n as i64;
            for (ch, y) in ys.iter().enumerate() {
                for (i, &v) in y.iter().enumerate() {
                    let t = base + i as i64;
                    if (0..len as i64).contains(&t) {
                        out[ch][t as usize] += v * ola_gain;
                    }
                }
            }
        }
        progress(f1, total);
        f0 = f1;
    }
    stats.mean_iterations = total_iters as f64 / stats.frames.max(1) as f64;
    for (dst, src) in channels.iter_mut().zip(out) {
        *dst = src;
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_factor_is_transparent() {
        let fs = 16_000;
        let x: Vec<f32> = (0..8000)
            .map(|i| {
                let t = i as f64 / fs as f64;
                (0.3 * (std::f64::consts::TAU * 150.0 * t).sin()
                    + 0.1 * (std::f64::consts::TAU * 1234.0 * t).sin()) as f32
            })
            .collect();
        let mut ch = vec![x.clone()];
        prewarp(&mut ch, fs, 1.0, 150.0, 2, &mut |_, _| {});
        let err = x
            .iter()
            .zip(&ch[0])
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 1e-4, "max error {err}");
    }

    #[test]
    fn silence_stays_silent() {
        let mut ch = vec![vec![0f32; 3000]; 2];
        let st = prewarp(&mut ch, 16_000, 1.5, 200.0, 2, &mut |_, _| {});
        assert_eq!(st.frames, 0);
        assert!(ch.iter().flatten().all(|&v| v == 0.0));
    }
}
