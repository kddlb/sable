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

/// Filters `channels` in place so their envelope reads `H(s·f)` at `f`.
pub fn prewarp(
    channels: &mut [Vec<f32>],
    fs: u32,
    s: f64,
    f0_ref_hz: f64,
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
    let mut fft = RealFft::new(k);
    let mut out = vec![vec![0f32; len]; nch];
    let mut frame = vec![0f32; k];
    let mut chan_spec: Vec<Vec<Complex>> = vec![vec![Complex::new(0.0, 0.0); bins]; nch];
    let mut log_mag = vec![0f64; bins];
    let mut a = vec![0f64; bins];
    let mut env = vec![0f64; bins];
    let mut scratch = vec![Complex::new(0.0, 0.0); bins];
    let mut mono_mag = vec![0f64; bins];
    let floor_ratio = 10f64.powf(FLOOR_DB / 20.0);
    let conv = CONVERGENCE_DB / 20.0 * std::f64::consts::LN_10;
    let max_gain = MAX_GAIN_DB / 20.0 * std::f64::consts::LN_10;
    let mut total_iters = 0usize;

    // Frame centres from −n/2 to len + n/2 so every sample gets the full window sum.
    let first = -(n as i64 / 2);
    let total = (len as i64 + n as i64 / 2 - first) as usize / hop + 1;
    for f in 0..total {
        let c = first + (f * hop) as i64;
        // Windowed frame in the middle of a 2n buffer: index i ↔ sample c − n + i.
        let mut any = false;
        mono_mag.fill(0.0);
        for ch in 0..nch {
            frame.fill(0.0);
            for (w_i, &w) in win.iter().enumerate() {
                let t = c - (n as i64) / 2 + w_i as i64;
                if (0..len as i64).contains(&t) {
                    let v = channels[ch][t as usize] * w;
                    any |= v != 0.0;
                    frame[n / 2 + w_i] = v;
                }
            }
            chan_spec[ch].copy_from_slice(fft.forward(&frame));
        }
        if !any {
            progress(f, total);
            continue;
        }
        // Envelope of the channel mean.
        for (b, m) in mono_mag.iter_mut().enumerate() {
            let sum: Complex = chan_spec.iter().map(|s| s[b]).sum();
            *m = (sum.norm() / nch as f32) as f64;
        }
        let peak = mono_mag.iter().fold(0f64, |p, &v| p.max(v));
        let floor = (peak * floor_ratio).max(1e-30);
        for (l, &m) in log_mag.iter_mut().zip(&mono_mag) {
            *l = (m + floor).ln();
        }
        a.copy_from_slice(&log_mag);
        let mut iters = 0;
        loop {
            cepstral_smooth(&mut fft, &a, order, &mut scratch, &mut env);
            iters += 1;
            let gap = log_mag
                .iter()
                .zip(&env)
                .fold(f64::NEG_INFINITY, |g, (&l, &e)| g.max(l - e));
            if gap < conv || iters >= MAX_ITERATIONS {
                break;
            }
            for (ai, &e) in a.iter_mut().zip(&env) {
                *ai = ai.max(e);
            }
        }
        total_iters += iters;
        stats.frames += 1;
        // Gain G(f) = H(s·f) / H(f), clamped; bins mapped past Nyquist read H(Nyquist).
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
            for spec in chan_spec.iter_mut() {
                spec[b] *= g;
            }
        }
        for ch in 0..nch {
            let y = fft.inverse(&chan_spec[ch]);
            let base = c - n as i64;
            for (i, &v) in y.iter().enumerate() {
                let t = base + i as i64;
                if (0..len as i64).contains(&t) {
                    out[ch][t as usize] += v * ola_gain;
                }
            }
        }
        progress(f, total);
    }
    progress(total, total);
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
        prewarp(&mut ch, fs, 1.0, 150.0, &mut |_, _| {});
        let err = x
            .iter()
            .zip(&ch[0])
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 1e-4, "max error {err}");
    }

    #[test]
    fn silence_stays_silent() {
        let mut ch = vec![vec![0f32; 3000]; 2];
        let st = prewarp(&mut ch, 16_000, 1.5, 200.0, &mut |_, _| {});
        assert_eq!(st.frames, 0);
        assert!(ch.iter().flatten().all(|&v| v == 0.0));
    }
}
