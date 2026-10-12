//! Shared acoustic analysis on an analysis-only mono reference (spec §4.1).
//!
//! All windows/hops are specified in milliseconds and converted per sample rate.
//! Frame `j` is centred at sample `j * hop + hop / 2`.

use crate::fft::RealFft;
use crate::windows::{hann_periodic, ms_to_samples, pow2_near_ms};

const EPS: f64 = 1e-20;

#[derive(Clone, Debug)]
pub struct FeatureConfig {
    pub hop_ms: f64,
    pub rms_ms: f64,
    pub fft_ms: f64,
    pub periodicity_ms: f64,
    pub f0_min_hz: f64,
    pub f0_max_hz: f64,
    pub hf_edge_hz: f64,
    pub low_band_hz: f64,
    pub fine_block_ms: f64,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            hop_ms: 5.0,
            rms_ms: 20.0,
            fft_ms: 25.0,
            periodicity_ms: 40.0,
            f0_min_hz: 65.0,
            f0_max_hz: 450.0,
            hf_edge_hz: 2500.0,
            low_band_hz: 1500.0,
            fine_block_ms: 1.0,
        }
    }
}

/// Frame-level features plus a 1-ms pre-emphasised envelope for transient timing.
#[derive(Clone, Debug)]
pub struct Features {
    pub fs: u32,
    pub n: usize,
    pub hop: usize,
    pub nfft: usize,
    /// Analysis-only mono reference (never rendered).
    pub mono: Vec<f32>,
    /// Frame RMS in dB (20 ms rectangular window).
    pub log_rms: Vec<f32>,
    /// Positive spectral flux normalized by local magnitude sum, in `[0, 1]`.
    pub flux: Vec<f32>,
    /// Energy above the adaptive high-band edge divided by energy above 80 Hz.
    pub hf_ratio: Vec<f32>,
    /// Energy-weighted local (≈500 Hz sub-band) spectral flatness of the high band.
    pub flatness: Vec<f32>,
    pub centroid_hz: Vec<f32>,
    /// Max normalized autocorrelation in the F0 lag range (full band).
    pub periodicity: Vec<f32>,
    /// Same, restricted to the band below `low_band_hz` (voiced-fricative evidence).
    pub periodicity_low: Vec<f32>,
    /// Fraction of energy below `low_band_hz`.
    pub low_frac: Vec<f32>,
    /// High-band energy relative to energy above `low_band_hz` (ignores a voiced low band).
    pub hf_ratio_upper: Vec<f32>,
    /// F0 estimate in Hz (0 where unvoiced).
    pub f0_hz: Vec<f32>,
    pub zcr: Vec<f32>,
    pub noise_floor_db: f32,
    pub speech_level_db: f32,
    pub silence_thr_db: f32,
    /// High-band edge actually used (Hz).
    pub hf_edge_hz: f64,
    pub fine_block: usize,
    /// Pre-emphasised 1-ms block energy in dB.
    pub fine_db: Vec<f32>,
}

impl Features {
    pub fn frames(&self) -> usize {
        self.log_rms.len()
    }

    pub fn frame_center(&self, j: usize) -> usize {
        j * self.hop + self.hop / 2
    }

    pub fn frame_of(&self, sample: usize) -> usize {
        (sample / self.hop).min(self.frames().saturating_sub(1))
    }

    pub fn is_silent(&self, j: usize) -> bool {
        self.log_rms[j] < self.silence_thr_db
    }
}

/// Builds the analysis-only mono reference: `0.5 (L + R)`, or the louder channel when
/// the mid signal loses more than 90 % of the mean channel energy (L/R cancellation).
pub fn mono_reference(channels: &[&[f32]]) -> Vec<f32> {
    match channels.len() {
        0 => Vec::new(),
        1 => channels[0].to_vec(),
        _ => {
            let (l, r) = (channels[0], channels[1]);
            let mid: Vec<f32> = l.iter().zip(r).map(|(a, b)| 0.5 * (a + b)).collect();
            let e = |x: &[f32]| x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
            let (el, er, em) = (e(l), e(r), e(&mid));
            if em < 0.1 * 0.5 * (el + er) {
                if el >= er { l.to_vec() } else { r.to_vec() }
            } else {
                mid
            }
        }
    }
}

fn read(x: &[f32], i: isize) -> f32 {
    if i < 0 || i as usize >= x.len() {
        0.0
    } else {
        x[i as usize]
    }
}

fn percentile(v: &[f32], p: f64) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(f32::total_cmp);
    let idx = ((s.len() - 1) as f64 * p).round() as usize;
    s[idx]
}

/// Computes all features for the mono reference of `channels`.
pub fn analyze(channels: &[&[f32]], fs: u32, cfg: &FeatureConfig) -> Features {
    let mono = mono_reference(channels);
    let n = mono.len();
    let hop = ms_to_samples(cfg.hop_ms, fs);
    let nf = n.div_ceil(hop).max(1);
    let rms_len = ms_to_samples(cfg.rms_ms, fs);
    let nfft = pow2_near_ms(cfg.fft_ms, fs);
    let bins = nfft / 2 + 1;
    let bin_hz = fs as f64 / nfft as f64;
    let nyq = fs as f64 / 2.0;
    let hf_edge = cfg.hf_edge_hz.min(0.6 * nyq);
    let k_hf = ((hf_edge / bin_hz).round() as usize).min(bins - 1);
    let k_low = ((80.0 / bin_hz).ceil() as usize).max(1);
    let k_lowband = ((cfg.low_band_hz / bin_hz).round() as usize).min(bins - 1);

    let mut log_rms = vec![0f32; nf];
    let mut flux = vec![0f32; nf];
    let mut hf_ratio = vec![0f32; nf];
    let mut flatness = vec![0f32; nf];
    let mut centroid = vec![0f32; nf];
    let mut zcr = vec![0f32; nf];
    let mut low_frac = vec![0f32; nf];
    let mut hf_ratio_upper = vec![0f32; nf];

    // Running prefix sums for RMS.
    let mut pre = vec![0f64; n + 1];
    for i in 0..n {
        pre[i + 1] = pre[i] + (mono[i] as f64) * (mono[i] as f64);
    }

    let win = hann_periodic(nfft);
    let mut fft = RealFft::new(nfft);
    let mut buf = vec![0f32; nfft];
    let mut prev_mag = vec![0f32; bins];
    let mut pw = vec![0f64; bins];
    let flat_band = ((500.0 / bin_hz).round() as usize).max(4);
    let mut mag = vec![0f32; bins];
    for j in 0..nf {
        let c = (j * hop + hop / 2) as isize;
        // RMS
        let a = (c - rms_len as isize / 2).clamp(0, n as isize) as usize;
        let b = (c + rms_len as isize / 2).clamp(0, n as isize) as usize;
        let e = if b > a {
            (pre[b] - pre[a]) / rms_len as f64
        } else {
            0.0
        };
        log_rms[j] = (10.0 * (e + EPS).log10()) as f32;
        // ZCR
        let mut zc = 0;
        for i in a.max(1)..b {
            if (mono[i] >= 0.0) != (mono[i - 1] >= 0.0) {
                zc += 1;
            }
        }
        zcr[j] = if b > a {
            zc as f32 / (b - a) as f32
        } else {
            0.0
        };
        // Spectrum
        let start = c - nfft as isize / 2;
        for (k, v) in buf.iter_mut().enumerate() {
            *v = read(&mono, start + k as isize);
        }
        let spec = fft.forward_windowed(&buf, &win);
        let (mut tot, mut hf, mut low, mut cen, mut lsum) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for k in 0..bins {
            let p = spec[k].norm_sqr() as f64;
            mag[k] = p.sqrt() as f32;
            pw[k] = p;
            if k >= k_low {
                tot += p;
                cen += p * k as f64 * bin_hz;
            }
            if k >= k_hf {
                hf += p;
            }
            if k < k_lowband {
                low += p;
            }
            lsum += p;
        }
        hf_ratio[j] = (hf / (tot + EPS)) as f32;
        flatness[j] = local_flatness(&pw[k_hf..], flat_band) as f32;
        centroid[j] = (cen / (tot + EPS)) as f32;
        low_frac[j] = (low / (lsum + EPS)) as f32;
        let upper: f64 = pw[k_lowband..].iter().sum();
        hf_ratio_upper[j] = (hf / (upper + EPS)) as f32;
        let (mut up, mut msum) = (0f64, 0f64);
        for k in 0..bins {
            up += (mag[k] - prev_mag[k]).max(0.0) as f64;
            msum += (mag[k] + prev_mag[k]) as f64;
        }
        flux[j] = if msum > 1e-12 {
            (up / msum) as f32 * 2.0
        } else {
            0.0
        };
        flux[j] = flux[j].min(1.0);
        std::mem::swap(&mut prev_mag, &mut mag);
    }

    let noise_floor_db = percentile(&log_rms, 0.10);
    let speech_level_db = percentile(&log_rms, 0.95);
    let range = speech_level_db - noise_floor_db;
    let silence_thr_db = if range < 15.0 {
        speech_level_db - 35.0
    } else {
        noise_floor_db + (0.3 * range).min(12.0)
    }
    .max(-120.0);

    let silent: Vec<bool> = log_rms.iter().map(|&v| v < silence_thr_db).collect();
    let (periodicity, periodicity_low, f0_hz) = periodicity_track(&mono, fs, hop, nf, cfg, &silent);

    // 1-ms pre-emphasised envelope.
    let fine_block = ms_to_samples(cfg.fine_block_ms, fs);
    let nb = n.div_ceil(fine_block);
    let mut fine_db = vec![0f32; nb];
    for (m, out) in fine_db.iter_mut().enumerate() {
        let a = m * fine_block;
        let b = (a + fine_block).min(n);
        let mut acc = 0f64;
        for i in a..b {
            let d = mono[i] as f64 - 0.97 * read(&mono, i as isize - 1) as f64;
            acc += d * d;
        }
        *out = (10.0 * (acc / fine_block as f64 + EPS).log10()) as f32;
    }

    Features {
        fs,
        n,
        hop,
        nfft,
        mono,
        log_rms,
        flux,
        hf_ratio,
        flatness,
        centroid_hz: centroid,
        periodicity,
        periodicity_low,
        low_frac,
        hf_ratio_upper,
        f0_hz,
        zcr,
        noise_floor_db,
        speech_level_db,
        silence_thr_db,
        hf_edge_hz: hf_edge,
        fine_block,
        fine_db,
    }
}

/// Energy-weighted mean of the spectral flatness measured in consecutive sub-bands of
/// `band` bins. A Gaussian-noise periodogram scores ≈ e^-γ ≈ 0.56 in every band regardless
/// of the overall spectral shape; harmonic spectra score much lower.
fn local_flatness(p: &[f64], band: usize) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for chunk in p.chunks(band) {
        if chunk.len() < band / 2 {
            break;
        }
        let am = chunk.iter().sum::<f64>() / chunk.len() as f64;
        if am <= EPS {
            continue;
        }
        let gm =
            (chunk.iter().map(|&v| (v + am * 1e-6).ln()).sum::<f64>() / chunk.len() as f64).exp();
        num += am * (gm / am);
        den += am;
    }
    if den > 0.0 { num / den } else { 0.0 }
}

/// Autocorrelation pitch tracker (Boersma-style window-normalized ACF via FFT).
fn periodicity_track(
    x: &[f32],
    fs: u32,
    hop: usize,
    nf: usize,
    cfg: &FeatureConfig,
    silent: &[bool],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut est = PitchEstimator::new(fs, cfg);
    let mut per = vec![0f32; nf];
    let mut per_low = vec![0f32; nf];
    let mut f0 = vec![0f32; nf];
    for j in 0..nf {
        if silent[j] {
            continue;
        }
        let c = j * hop + hop / 2;
        let (p, f, pl) = est.estimate(x, c);
        per[j] = p;
        f0[j] = if p > 0.45 { f } else { 0.0 };
        per_low[j] = pl;
    }
    (per, per_low, f0)
}

/// Reusable normalized-autocorrelation pitch estimator.
pub struct PitchEstimator {
    fs: u32,
    w: usize,
    win: Vec<f32>,
    rw: Vec<f64>,
    fft: RealFft,
    lag_min: usize,
    lag_max: usize,
    k_low: usize,
    buf: Vec<f32>,
}

impl PitchEstimator {
    pub fn new(fs: u32, cfg: &FeatureConfig) -> Self {
        let w = ms_to_samples(cfg.periodicity_ms, fs);
        let nfft = (2 * w).next_power_of_two();
        let win: Vec<f32> = crate::windows::hann_offset(w);
        let mut fft = RealFft::new(nfft);
        // Window autocorrelation for normalization.
        let mut b = vec![0f32; nfft];
        b[..w].copy_from_slice(&win);
        let spec: Vec<_> = fft.forward(&b).iter().map(|c| c.norm_sqr()).collect();
        let cs: Vec<_> = spec
            .iter()
            .map(|&p| crate::fft::Complex::new(p, 0.0))
            .collect();
        let r = fft.inverse(&cs).to_vec();
        let r0 = r[0].max(1e-12) as f64;
        let rw = r.iter().map(|&v| v as f64 / r0).collect();
        let lag_min = ((fs as f64 / cfg.f0_max_hz).floor() as usize).max(2);
        let lag_max = ((fs as f64 / cfg.f0_min_hz).ceil() as usize).min(w / 2);
        let k_low = ((cfg.low_band_hz * nfft as f64 / fs as f64).round() as usize).min(nfft / 2);
        Self {
            fs,
            w,
            win,
            rw,
            fft,
            lag_min,
            lag_max,
            k_low,
            buf: vec![0.0; nfft],
        }
    }

    /// Returns `(periodicity, f0_hz, low_band_periodicity)` for a window centred at `c`.
    pub fn estimate(&mut self, x: &[f32], c: usize) -> (f32, f32, f32) {
        let start = c as isize - self.w as isize / 2;
        let mut mean = 0f64;
        for k in 0..self.w {
            mean += read(x, start + k as isize) as f64;
        }
        mean /= self.w as f64;
        self.buf.iter_mut().for_each(|v| *v = 0.0);
        for k in 0..self.w {
            self.buf[k] = (read(x, start + k as isize) as f64 - mean) as f32 * self.win[k];
        }
        let pw: Vec<f32> = self
            .fft
            .forward(&self.buf)
            .iter()
            .map(|c| c.norm_sqr())
            .collect();
        let full: Vec<_> = pw
            .iter()
            .map(|&p| crate::fft::Complex::new(p, 0.0))
            .collect();
        let r = self.fft.inverse(&full).to_vec();
        let (p, lag) = self.best_peak(&r);
        let low: Vec<_> = pw
            .iter()
            .enumerate()
            .map(|(k, &p)| crate::fft::Complex::new(if k <= self.k_low { p } else { 0.0 }, 0.0))
            .collect();
        let rl = self.fft.inverse(&low).to_vec();
        let (pl, _) = self.best_peak(&rl);
        let f0 = if lag > 0.0 {
            self.fs as f32 / lag as f32
        } else {
            0.0
        };
        (p as f32, f0, pl as f32)
    }

    fn best_peak(&self, r: &[f32]) -> (f64, f64) {
        let r0 = r[0] as f64;
        if r0 <= 1e-12 {
            return (0.0, 0.0);
        }
        let norm = |t: usize| (r[t] as f64 / r0) / self.rw[t].max(1e-3);
        let mut best = (0.0f64, 0usize);
        for t in self.lag_min..=self.lag_max {
            let v = norm(t);
            // Local maximum only, with a tiny bias toward shorter lags (octave errors).
            if v > norm(t - 1) && v >= norm(t + 1) {
                let biased = v - 0.02 * (t as f64 / self.lag_min as f64).log2();
                let best_b = if best.1 == 0 {
                    f64::NEG_INFINITY
                } else {
                    best.0 - 0.02 * (best.1 as f64 / self.lag_min as f64).log2()
                };
                if biased > best_b {
                    best = (v, t);
                }
            }
        }
        if best.1 == 0 {
            return (0.0, 0.0);
        }
        let t = best.1;
        let (a, b, c) = (norm(t - 1), norm(t), norm(t + 1));
        let den = a - 2.0 * b + c;
        let delta = if den.abs() > 1e-12 {
            (0.5 * (a - c) / den).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        (best.0.clamp(0.0, 1.0), t as f64 + delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pitch_of_sine_mixture() {
        let fs = 16_000;
        let f0 = 140.0;
        let x: Vec<f32> = (0..fs as usize)
            .map(|i| {
                let t = i as f64 / fs as f64;
                (0..8)
                    .map(|h| {
                        (std::f64::consts::TAU * f0 * (h + 1) as f64 * t).sin() / (h + 1) as f64
                    })
                    .sum::<f64>() as f32
                    * 0.3
            })
            .collect();
        let mut pe = PitchEstimator::new(fs, &FeatureConfig::default());
        let (p, f, _) = pe.estimate(&x, 8000);
        assert!(p > 0.9, "periodicity {p}");
        assert!((f as f64 - f0).abs() / f0 < 0.01, "f0 {f}");
    }

    #[test]
    fn silence_features_are_finite() {
        let z = vec![0f32; 4000];
        let f = analyze(&[&z], 8000, &FeatureConfig::default());
        assert!(f.log_rms.iter().all(|v| v.is_finite()));
        assert!(f.flux.iter().all(|v| *v == 0.0));
        assert!(f.periodicity.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn cancelling_stereo_uses_louder_channel() {
        let l: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let r: Vec<f32> = l.iter().map(|v| -v * 0.9).collect();
        let m = mono_reference(&[&l, &r]);
        assert_eq!(m, l);
    }
}
